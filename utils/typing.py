import sys
import os
import functools
import numpy as np
import tensorflow as tf
from mpi4py import MPI

class TypeUtils:

    def list_like(data):
        if isinstance(data, range):
            list_flag = True
            ret_type = range
        elif isinstance(data, tuple):
            list_flag = True
            ret_type = tuple
        elif isinstance(data, np.ndarray):
            list_flag = True
            ret_type = np.ndarray
        elif isinstance(data, list):
            list_flag = True
            ret_type = list
        else:
            list_flag = False
            ret_type = type(data)
        return list_flag, ret_type

    def get_flat_size(data):
        list_flag, _ = TypeUtils.list_like(data)
        if not list_flag:
            return 1
        return sum([TypeUtils.get_flat_size(x) for x in data])

    # def get_shape(data):
    #     list_flag, _ = TypeUtils.list_like(data)
    #     if not list_flag:
    #         return type(data)
    #     return [TypeUtils.get_shape(x) for x in data]

    def get_shape(data):
        list_flag, data_type = TypeUtils.list_like(data)
        if isinstance(data, np.ndarray):
            return data.shape
        elif not list_flag:
            return 1
        return [TypeUtils.get_shape(x) for x in data]

    def get_type(data, cast=None):
        list_flag, data_type = TypeUtils.list_like(data)
        if isinstance(data, np.ndarray):
            if cast is not None:
                return cast(data.dtype)
            return data.dtype
        elif not list_flag:
            if cast is not None:
                return cast(data_type)
            return data_type
        return [TypeUtils.get_type(x, cast=cast) for x in data]

    def get_dumps(data):
        list_flag, _ = TypeUtils.list_like(data)
        if not list_flag:
            return len(MPI.pickle.dumps(data))
        return [TypeUtils.get_dumps(x) for x in data]

    def check_diff(data1, data2):
        list_flag1, the_type1 = TypeUtils.list_like(data1)
        list_flag2, the_type2 = TypeUtils.list_like(data2)

        if the_type1 != the_type2:
            print("List Type Mismatch", the_type1, the_type2)
            return True

        if list_flag1 and list_flag2:
            for i, j in zip(data1, data2):
                return TypeUtils.check_diff(i, j)

        elif not list_flag1 and not list_flag2:
            if data1 != data2:
                print("Mismatch:", data1, "!=", data2)
                return True

        else:
            print("List Mismatch:", the_type1, "!=", the_type2)
            return True
        return False

    def compare(data1, data2):
        ret = True
        data1_size = TypeUtils.get_flat_size(data1)
        data2_size = TypeUtils.get_flat_size(data2)
        if data1_size != data2_size:
            print("Elem Size Error:", data1_size, data2_size)
            print("Data 1:", TypeUtils.get_shape(data1))
            print("Data 2:", TypeUtils.get_shape(data2))
            return False

        data1_shape = TypeUtils.get_shape(data1)
        data2_shape = TypeUtils.get_shape(data2)
        if TypeUtils.check_diff(data1_shape, data2_shape):
            print("Shape Error")
            print("Data 1:", data1_shape)
            print("Data 2:", data2_shape)
            return False

        data1_dump_len = len(MPI.pickle.dumps(data1))
        data2_dump_len = len(MPI.pickle.dumps(data2))
        if data1_dump_len != data2_dump_len:
            print("Dump Error", data1_dump_len, data2_dump_len)
            data1_dumps = TypeUtils.get_dumps(data1)
            data2_dumps = TypeUtils.get_dumps(data2)
            if TypeUtils.check_diff(data1_dumps, data2_dumps):
                print("Data 1:", data1_dumps)
                print("Data 2:", data2_dumps)
            return False
        return True

    def np_type_converter(the_type, promote=True):
        if the_type == float or the_type == np.float64 or the_type == tf.float64 or the_type == MPI.DOUBLE:
            return np.float64
        if the_type == np.float32 or the_type == tf.float32 or the_type == MPI.FLOAT:
            if promote:
                return np.float64
            return np.float32
        if the_type == int or the_type == np.int64 or the_type == tf.int64 or the_type == MPI.INT64_T:
            return np.int64
        if the_type == np.int32 or the_type == tf.int32 or the_type == MPI.INT:
            if promote:
                return np.int64
            return np.int32
        if the_type == bool or the_type == np.bool or the_type == tf.bool or the_type == MPI.BOOL:
            return np.bool
        print("Failed to convert type", the_type, "to np type")
        return the_type

    def tf_type_converter(the_type, promote=True):
        if the_type == float or the_type == np.float64 or the_type == tf.float64 or the_type == MPI.DOUBLE:
            return tf.float64
        if the_type == np.float32 or the_type == tf.float32 or the_type == MPI.FLOAT:
            if promote:
                return tf.float64
            return tf.float32
        if the_type == int or the_type == np.int64 or the_type == tf.int64 or the_type == MPI.INT64_T:
            return tf.int64
        if the_type == np.int32 or the_type == tf.int32 or the_type == MPI.INT:
            if promote:
                return tf.int64
            return tf.int32
        if the_type == bool or the_type == np.bool or the_type == tf.bool or the_type == MPI.BOOL:
            return tf.bool
        print("Failed to convert type", the_type, "to tf type")
        return the_type

    def promote_numpy_type(data, makeList=True):
        list_flag, the_type = TypeUtils.list_like(data)
        if not list_flag and makeList:
            return np.array([data], dtype=TypeUtils.np_type_converter(the_type))

        if isinstance(data, np.ndarray):
            if data.dtype == np.float32:
                return data.astype(np.float64)
            elif data.dtype == np.int32:
                return data.astype(np.int64)
        return data
