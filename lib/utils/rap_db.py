import cv2
import scipy.io as sio
import os.path as osp


class RAP_DB:
	def __init__(self, db_path, par_set_id):
		self._db_path = db_path

		rap = sio.loadmat(osp.join(self._db_path,'RAP_annotation','RAP_annotation.mat'))['RAP_annotation']

		self._partition = rap[0][0][0]
		self.labels = rap[0][0][1]
		self.attr_ch = rap[0][0][2]
		self.attr_eng = rap[0][0][3]
		self.position = rap[0][0][4]
		self.img_names = rap[0][0][5]
		self.attr_exp = rap[0][0][6]

		self.set_partition_set_id(par_set_id)

	def load_image(self, img_name):
		return cv2.imread(osp.join(self._db_path, 'RAP_dataset', img_name))

	def set_partition_set_id(self, par_set_id):
		self.train_ind = self._partition[par_set_id][0][0][0][0][0] - 1
		self.test_ind = self._partition[par_set_id][0][0][0][1][0] - 1


if __name__ == '__main__':
	db = RAP_DB('/home/ken.yu/datasets/rap', 0)
	print db._partition.shape
	print db._partition[0][0][0][0][1].shape
	print db._partition[1][0][0][0][1].shape
	print db.labels.shape
	print db.img_names.shape
	print db.train_ind.shape
	print 'Max training index: ', max(db.train_ind)
	print db.img_names[db.train_ind]