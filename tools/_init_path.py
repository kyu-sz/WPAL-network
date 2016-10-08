#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of WMA Network.
#
# WMA Network is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WMA Network is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with WMA Network.  If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

"""Set up paths for AM Net."""

import os
import os.path as osp
import sys


def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
project_path = osp.join(this_dir, '..')
add_path(project_path)

# Add caffe to PYTHONPATH
caffe_path = osp.join(project_path, 'caffe', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(project_path, 'lib')
add_path(lib_path)

os.chdir(project_path)
