"""Candidate seven-hand FAAS codec, full state and control targets kept distinct."""
import numpy as np


class FAAS7:
    def __init__(self,contract):
        self.nj=len(contract['joint_names']);self.na=len(contract['actuator_names'])
        for key in ('full_to_control','control_to_full','mimic_bias','faas_slots','faas_scale','faas_offset'):
            setattr(self,key,np.asarray(contract[key]))
        self.mask=np.zeros(32,bool);self.mask[self.faas_slots]=True

    @staticmethod
    def checked(value,n):
        value=np.asarray(value)
        if value.ndim<1 or value.shape[-1]!=n or not np.isfinite(value).all():
            raise ValueError(f'expected finite (...,{n})')
        return value

    def state(self,full_q):
        q=self.checked(full_q,self.nj)
        z=np.zeros(q.shape[:-1]+(32,),dtype=np.result_type(q,np.float64))
        z[...,self.faas_slots]=q*self.faas_scale+self.faas_offset
        return z

    def action(self,control):
        control=self.checked(control,self.na)
        full_q=control@self.control_to_full.T+self.mimic_bias
        return self.state(full_q)

    def decode(self,z):
        z=self.checked(z,32)
        q=(z[...,self.faas_slots]-self.faas_offset)/self.faas_scale
        return q@self.full_to_control.T

    def decode_state(self,z):
        z=self.checked(z,32)
        return (z[...,self.faas_slots]-self.faas_offset)/self.faas_scale
