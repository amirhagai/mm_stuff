# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import mmengine
from mmengine.fileio import HTTPBackend, PetrelBackend

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()

test_data_dir = osp.dirname(osp.dirname(__file__))


def _test_handler(file_format, test_obj, str_checker, mode='r+'):
    # dump to a string
    dump_str = mmengine.dump(test_obj, file_format=file_format)
    str_checker(dump_str)

    # load/dump with filenames from disk
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmengine_test_dump')
    mmengine.dump(test_obj, tmp_filename, file_format=file_format)
    assert osp.isfile(tmp_filename)
    load_obj = mmengine.load(tmp_filename, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # load/dump with filename from petrel
    method = 'put' if 'b' in mode else 'put_text'
    with patch.object(PetrelBackend, method, return_value=None) as mock_method:
        filename = 's3://path/of/your/file'
        mmengine.dump(test_obj, filename, file_format=file_format)
    mock_method.assert_called()

    # json load/dump with a file-like object
    with tempfile.NamedTemporaryFile(mode, delete=False) as f:
        tmp_filename = f.name
        mmengine.dump(test_obj, f, file_format=file_format)
    assert osp.isfile(tmp_filename)
    with open(tmp_filename, mode) as f:
        load_obj = mmengine.load(f, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # automatically inference the file format from the given filename
    tmp_filename = osp.join(tempfile.gettempdir(),
                            'mmengine_test_dump.' + file_format)
    mmengine.dump(test_obj, tmp_filename)
    assert osp.isfile(tmp_filename)
    load_obj = mmengine.load(tmp_filename)
    assert load_obj == test_obj
    os.remove(tmp_filename)


obj_for_test = [{'a': 'abc', 'b': 1}, 2, 'c']


def test_json():

    def json_checker(dump_str):
        assert dump_str in [
            '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
        ]

    _test_handler('json', obj_for_test, json_checker)


def test_yaml():

    def yaml_checker(dump_str):
        assert dump_str in [
            '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n',
            '- a: abc\n  b: 1\n- 2\n- c\n', '- b: 1\n  a: abc\n- 2\n- c\n'
        ]

    _test_handler('yaml', obj_for_test, yaml_checker)


def test_pickle():

    def pickle_checker(dump_str):
        import pickle
        assert pickle.loads(dump_str) == obj_for_test

    _test_handler('pickle', obj_for_test, pickle_checker, mode='rb+')


def test_exception():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']

    with pytest.raises(ValueError):
        mmengine.dump(test_obj)

    with pytest.raises(TypeError):
        mmengine.dump(test_obj, 'tmp.txt')


def test_register_handler():

    @mmengine.register_handler('txt')
    class TxtHandler1(mmengine.BaseFileHandler):

        def load_from_fileobj(self, file):
            return file.read()

        def dump_to_fileobj(self, obj, file):
            file.write(str(obj))

        def dump_to_str(self, obj, **kwargs):
            return str(obj)

    @mmengine.register_handler(['txt1', 'txt2'])
    class TxtHandler2(mmengine.BaseFileHandler):

        def load_from_fileobj(self, file):
            return file.read()

        def dump_to_fileobj(self, obj, file):
            file.write('\n')
            file.write(str(obj))

        def dump_to_str(self, obj, **kwargs):
            return str(obj)

    content = mmengine.load(osp.join(test_data_dir, 'data/filelist.txt'))
    assert content == '1.jpg\n2.jpg\n3.jpg\n4.jpg\n5.jpg'
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmengine_test.txt2')
    mmengine.dump(content, tmp_filename)
    with open(tmp_filename) as f:
        written = f.read()
    os.remove(tmp_filename)
    assert written == '\n' + content


def test_list_from_file():
    # get list from disk
    filename = osp.join(test_data_dir, 'data/filelist.txt')
    filelist = mmengine.list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = mmengine.list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = mmengine.list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = mmengine.list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = mmengine.list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']

    # get list from http
    filename = 'http://path/of/your/file'
    with patch.object(
            HTTPBackend, 'get_text', return_value='1.jpg\n2.jpg\n3.jpg'):
        filelist = mmengine.list_from_file(
            filename, file_client_args={'backend': 'http'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = mmengine.list_from_file(
            filename, file_client_args={'prefix': 'http'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']

        filelist = mmengine.list_from_file(filename)
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = mmengine.list_from_file(
            filename, backend_args={'backend': 'http'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']

    # get list from petrel
    filename = 's3://path/of/your/file'
    with patch.object(
            PetrelBackend, 'get_text', return_value='1.jpg\n2.jpg\n3.jpg'):
        filelist = mmengine.list_from_file(
            filename, file_client_args={'backend': 'petrel'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = mmengine.list_from_file(
            filename, file_client_args={'prefix': 's3'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = mmengine.list_from_file(filename)
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = mmengine.list_from_file(
            filename, backend_args={'backend': 'petrel'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']


def test_dict_from_file():
    # get dict from disk
    filename = osp.join(test_data_dir, 'data/mapping.txt')
    mapping = mmengine.dict_from_file(filename)
    assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
    mapping = mmengine.dict_from_file(filename, key_type=int)
    assert mapping == {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}

    # get dict from http
    filename = 'http://path/of/your/file'
    with patch.object(
            HTTPBackend, 'get_text', return_value='1 cat\n2 dog cow\n3 panda'):
        mapping = mmengine.dict_from_file(
            filename, file_client_args={'backend': 'http'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = mmengine.dict_from_file(
            filename, file_client_args={'prefix': 'http'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}

        mapping = mmengine.dict_from_file(filename)
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = mmengine.dict_from_file(
            filename, backend_args={'backend': 'http'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}

    # get dict from petrel
    filename = 's3://path/of/your/file'
    with patch.object(
            PetrelBackend, 'get_text',
            return_value='1 cat\n2 dog cow\n3 panda'):
        mapping = mmengine.dict_from_file(
            filename, file_client_args={'backend': 'petrel'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = mmengine.dict_from_file(
            filename, file_client_args={'prefix': 's3'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}

        mapping = mmengine.dict_from_file(filename)
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = mmengine.dict_from_file(
            filename, backend_args={'backend': 'petrel'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
