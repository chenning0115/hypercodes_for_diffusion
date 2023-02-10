import os, sys
import json, time


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def get_avg(self):
        return self.avg

    def get_num(self):
        return self.cnt
    
class HSIRecoder(object):
    def __init__(self) -> None:
        self.record_data = {}

    def append_index_value(self, name, index, value):
        """
        index : int, 
        value: Any
        save to dict
        {index: list, value: list}
        """
        if name not in self.record_data:
            self.record_data[name] = {
                "type": "index_value",
                "index":[],
                "value":[]
            } 
        self.record_data[name]['index'].append(index)
        self.record_data[name]['value'].append(value)

    def record_param(self, param):
        self.record_data['param'] = param 

    def record_eval(self, eval_obj):
        self.record_data['eval'] = eval_obj
        
    def to_file(self, path):
        time_stamp = int(time.time())
        save_path = "%s_%s.json" % (path, str(time_stamp))
        ss = json.dumps(self.record_data, indent=4)
        with open(save_path, 'w') as fout:
            fout.write(ss)
            fout.flush()

    def reset(self):
        self.record_data = {}
        

# global recorder
recorder = HSIRecoder()