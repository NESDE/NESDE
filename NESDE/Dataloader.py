import torch
import numpy as np


class Dataloader():
    def __init__(self, data, split, batch_size, shuffle, global_shuffle=True, verbose=False):
        # data - a list of trajectories
        # split - train/validation/test split ratio (i.e. [0.6,0.1,0.3]), set to False to disable
        # batch_size - obvious..
        # shuffle - shuffle data between epochs

        self.data = data
        self.batch_size = batch_size
        self.split = bool(split)
        self.shuffle = shuffle

        if split:
            if np.sum(split) <= 1:
                # input given as ratios - convert to number of trajectories
                split = [int(len(self.data) * p) for p in split]
            if np.sum(split) > len(self.data):
                raise ValueError(split, len(self.data))
            rand_idx = np.arange(len(self.data))
            if global_shuffle:
                np.random.shuffle(rand_idx)
            _min = [0,0,0]
            _max = [0,0,0]
            _min[0] = 0
            _max[0] = split[0]
            _min[1] = _max[0]
            _max[1] = split[1] + _min[1]
            _min[2] = len(self.data) - split[2]
            _max[2] = len(self.data)
            self.__ids = [rand_idx[np.arange(_min[0],_max[0])],rand_idx[np.arange(_min[1],_max[1])],rand_idx[np.arange(_min[2],_max[2])]]
            self.__it = [0,0,0]
            self.__sp_ind = 0
            if verbose:
                print(f'Dataloader - {len(self.data)} trajectories: {_min[0]}-{_max[0]} train, '
                      f'{_min[1]}-{_max[1]} valid, {_min[2]}-{_max[2]} test.')
        else:
            self.__it = 0
            self.__ids = np.arange(len(self.data))

    def it(self):
        if self.split:
            it = self.__it[self.__sp_ind]
            self.__it[self.__sp_ind] += self.batch_size
        else:
            it = self.__it
            self.__it += self.batch_size
        return it

    def ids(self, it):
        if self.split:
            return self.__ids[self.__sp_ind][it:it+self.batch_size]
        else:
            return self.__ids[it:it+self.batch_size]

    def train(self):
        self.__sp_ind = 0

    def valid(self):
        self.__sp_ind = 1

    def test(self):
        self.__sp_ind = 2

    def reset_it(self, all_i=False):
        if self.split:
            if all_i:
                self.__it = [0,0,0]
                if self.shuffle:
                    for i in range(len(self.__ids)):
                        np.random.shuffle(self.__ids[i])
            else:
                self.__it[self.__sp_ind] = 0
                if self.shuffle:
                    np.random.shuffle(self.__ids[self.__sp_ind])
        else:
            self.__it = 0
            if self.shuffle:
                np.random.shuffle(self.__ids)

    def epoch_done(self):
        if self.split:
            if self.__it[self.__sp_ind] + self.batch_size > len(self.__ids[self.__sp_ind]):
                self.reset_it()
                return True
            else:
                return False
        else:
            if self.__it + self.batch_size > len(self.__ids):
                self.reset_it()
                return True
            else:
                return False

    def get_batch(self, return_ids=False):
        it = self.it()
        data_l = [self.data[idx] for idx in self.ids(it)]
        # self.it += self.batch_size
        if return_ids:
            return data_l, self.epoch_done(), self.ids((it))
        else:
            return data_l, self.epoch_done()



# if __name__ == "__main__":
#     data = [np.ones(30) for i in range(110)]
#     dl = Dataloader(data=data,split=[0.6,0.1,0.3],batch_size=4, shuffle=False)
#
#     ldd = 0
#     dl.test()
#     for j in range(100):
#         batch, done = dl.get_batch()
#         ldd += len(batch)
#         if done:
#             print(ldd)
#             break
