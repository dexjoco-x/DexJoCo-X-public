"""Archive integrity and no-overwrite checks for the default dataset downloader."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools import training_packages as packages


class TrainingPackagesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {'bundle/catalog.json': b'{"dataset_version":"test"}\n',
                        'bundle/video.mp4': b'video', 'prepared/video.mp4': b'video'}
        self.index = dict(dataset_version='test', tasks={'hammer_nail': {}}, episodes=1,
                          frames=1, archive='data.tar', files=[
                              dict(path=k, bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                              for k, v in self.payload.items()])
        self.dest = self.root / 'out'

    def archive(self, special=None):
        path = self.root / 'data.tar'
        with tarfile.open(path, 'w') as out:
            rows = dict(self.payload, SHA256SUMS=packages.checksum_text(self.index).encode())
            for name, data in rows.items():
                member = tarfile.TarInfo(name)
                if name == 'prepared/video.mp4':
                    member.type = tarfile.LNKTYPE
                    member.linkname = 'bundle/video.mp4'
                    if special:
                        special(member)
                    out.addfile(member)
                else:
                    member.size = len(data)
                    out.addfile(member, io.BytesIO(data))
        self.index.update(bytes=path.stat().st_size, sha256=packages.sha(path))
        return path

    def test_exact_extraction_and_idempotent_reuse(self):
        path = self.archive()
        self.assertTrue(packages.extract(path, self.dest, self.index)['passed'])
        self.assertEqual((self.dest / 'prepared/video.mp4').read_bytes(), b'video')
        self.assertFalse((self.dest / 'prepared/video.mp4').is_symlink())
        self.assertTrue(packages.extract(path, self.dest, self.index)['passed'])

    def test_archive_digest_failure_leaves_no_destination(self):
        path = self.archive()
        self.index['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            packages.extract(path, self.dest, self.index)
        self.assertFalse(self.dest.exists())

    def test_member_digest_failure_leaves_no_destination(self):
        self.payload['bundle/video.mp4'] = b'wrong'
        path = self.archive()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            packages.extract(path, self.dest, self.index)
        self.assertFalse(self.dest.exists())

    def test_old_dataset_is_not_overwritten(self):
        path = self.archive()
        self.dest.mkdir()
        old = self.dest / 'old.txt'
        old.write_text('keep this')
        with self.assertRaises(FileExistsError):
            packages.extract(path, self.dest, self.index)
        self.assertEqual(old.read_text(), 'keep this')

    def test_symlink_parent_is_rejected(self):
        path = self.archive()
        actual = self.root / 'actual'
        actual.mkdir()
        self.dest.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlinks'):
            packages.extract(path, self.dest, self.index)
        self.assertEqual(list(actual.iterdir()), [])

    def test_unsafe_members_are_rejected(self):
        changes = [lambda m: setattr(m, 'type', tarfile.SYMTYPE),
                   lambda m: setattr(m, 'linkname', '../outside'),
                   lambda m: setattr(m, 'name', '../outside'),
                   lambda m: setattr(m, 'linkname', 'bundle/catalog.json')]
        for change in changes:
            with self.subTest(change=change):
                path = self.archive(change)
                with self.assertRaises(ValueError):
                    packages.extract(path, self.dest, self.index)
                self.assertFalse(self.dest.exists())


if __name__ == '__main__':
    unittest.main()
