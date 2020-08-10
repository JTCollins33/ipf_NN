import numpy as np
import math
import torchvision.transforms as transforms
import torch
import random

nPoints = 36

def check_append(full, part):
    add = True

    if (len(full) > 0):
        i = 0
        while (i < len(full)):
            if (full[i]==part[i%3] and full[i+1]==part[(i+1)%3] and full[i+2]==part[(i+2)%3]):
                add = False
            i+=3
    return add



def read_one_file(k, Dir):
    CAxis_file = Dir + "CAxisLocation_Image_"+str(k)+".txt"
    track_arr = np.array([])
    file = open(CAxis_file)

    i = 0
    for line in enumerate(file):
        line = line[1]
        if (line[0] != 'C'):
            current_arr = np.array([0.0, 0.0, 0.0])
            cnt = 0
            st = ""
            for i in range(0, len(line)):
                if (line[i] != ','):
                    st += str(line[i])

                elif (line[i] == ','):
                    current_arr[cnt] = float(st)
                    cnt+=1
                    st = ""

            current_arr[2] = st
            add = check_append(track_arr, current_arr)
            if(add):
                track_arr = np.append(track_arr, current_arr)    
    file.close()
    return track_arr

def read_one_file_test(k, Dir):
    CAxis_file = Dir + "CAxisLocation_Image_"+str(k)+".txt"
    track_arr = np.array([])
    file = open(CAxis_file)

    i = 0
    for line in enumerate(file):
        line = line[1]
        if (line[0] != 'C'):
            current_arr = np.array([0.0, 0.0, 0.0])
            cnt = 0
            st = ""
            for i in range(0, len(line)):
                if (line[i] != ','):
                    st += str(line[i])

                elif (line[i] == ','):
                    current_arr[cnt] = float(st)
                    cnt+=1
                    st = ""

            current_arr[2] = st
            track_arr = np.append(track_arr, current_arr)    
    file.close()
    return track_arr

def find_CAxis_Stats(f1, f2, f3):
    amp = int(100*(1-f3))
    v_shift = 0
    if (not(f1==0 and f2==0)):
        v1 = [0,1]
        v2 = [f1,f2]
        unit_v2 = v2/np.linalg.norm(v2)
        dot_product = np.dot(v1, unit_v2)
        angle = np.arccos(dot_product)
        v_shift = int((angle/(2.0*math.pi))*255)

        if (v_shift-amp < 0):
            v_shift = v_shift -(v_shift-amp)

    return amp, v_shift


def generate_sine_arrs(amp, v_shift):
    sine_arr = np.zeros(nPoints, dtype = np.float32)

    for i in range(1, nPoints+1):
        bias = 0
        rad = ((4*math.pi)/(1.0*nPoints))*i
        # sine_arr[i-1] = (amp*math.sin(rad)+v_shift)/(1.0*(amp+v_shift)) * 255
        if (amp==0 and v_shift==0):
            bias=1
        sine_arr[i-1] = ((amp*math.sin(rad)+v_shift)/(1.0*(amp+v_shift+bias))-0.5)*2

    return sine_arr

def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def convert_CAxis_to_Sine(Dir, to_tens):
    full_list = []
    full_CAxis = []
    for i in range(1, 501):
        CAxis_arr = read_one_file(i, Dir)
        sine_arr = np.zeros((int(len(CAxis_arr)/3), nPoints), dtype = np.float32)
        CAxis_arr2 = np.zeros((int(len(CAxis_arr)/3), 3), dtype = np.float32)
        j = 0
        while (j < len(CAxis_arr)):
            amp, v_shift = find_CAxis_Stats(CAxis_arr[j], CAxis_arr[j+1], CAxis_arr[j+2])
            sine_arr[int(j/3),:] = generate_sine_arrs(amp, v_shift)
            CAxis_arr2[int(j/3),:] = [CAxis_arr[j], CAxis_arr[j+1], CAxis_arr[j+2]]
            j+=3
            
        if (to_tens):
            tensor_transform = get_transform()
            sine_arr = tensor_transform(sine_arr)
            CAxis_arr2 = tensor_transform(CAxis_arr2)
        full_list.append([sine_arr, i])
        full_CAxis.append(CAxis_arr2)
    return full_list, full_CAxis

def convert_CAxis_to_Sine_test(Dir, to_tens):
    full_list = []
    full_CAxis = []

    for i in range(501, 551):
        CAxis_arr = read_one_file_test(i, Dir)
        sine_arr = np.zeros((int(len(CAxis_arr)/3), nPoints), dtype = np.float32)
        CAxis_arr2 = np.zeros((int(len(CAxis_arr)/3), 3), dtype = np.float32)
        j = 0
        while (j < len(CAxis_arr)):
            amp, v_shift = find_CAxis_Stats(CAxis_arr[j], CAxis_arr[j+1], CAxis_arr[j+2])
            sine_arr[int(j/3),:] = generate_sine_arrs(amp, v_shift)
            CAxis_arr2[int(j/3),:] = [CAxis_arr[j], CAxis_arr[j+1], CAxis_arr[j+2]]
            j+=3
            
        if (to_tens):
            tensor_transform = get_transform()
            sine_arr = tensor_transform(sine_arr)
            CAxis_arr2 = tensor_transform(CAxis_arr2)
        full_list.append([sine_arr, i])
        full_CAxis.append(CAxis_arr2)
    return full_list, full_CAxis

def get_sine_train_test_data(file_path, to_tens, n_train_samples, batch_size):
    sine_test = []
    sine_train = []
    CAxis_test = []
    CAxis_train = []

    line_cnt = 0
    batch_cnt = 0

    tensor_transform = get_transform()

    f = open(file_path, "r")

    current_batch_sine = np.zeros((batch_size, nPoints), dtype = np.float32)
    current_batch_CAxis = np.zeros((batch_size, 3), dtype = np.float32)
    for line in enumerate(f):
        triplet = line[1]
        if (triplet[0] != 'C'):
            line_cnt += 1
            current_triplet = np.array([0.0, 0.0, 0.0])
            cnt = 0
            st = ""
            for i in range(0, len(triplet)):
                if (triplet[i] != ',' and triplet[i] != '\n'):
                    st += str(triplet[i])

                elif (triplet[i] == ','):
                    current_triplet[cnt] = float(st)
                    cnt+=1
                    st = ""

            current_triplet[2] = float(st)

            #generate sine array from CAxis triplet
            amp, v_shift = find_CAxis_Stats(current_triplet[0], current_triplet[1], current_triplet[2])
            current_sine_arr = generate_sine_arrs(amp, v_shift)

            #if still in current batch, add data to batch
            if(batch_cnt < batch_size):
                current_batch_sine[batch_cnt] = current_sine_arr
                current_batch_CAxis[batch_cnt] = current_triplet

            #if batch is full, add back to full list and create new batch
            else:
                batch_cnt = 0
                current_batch_sine = tensor_transform(current_batch_sine)
                current_batch_CAxis = tensor_transform(current_batch_CAxis)

                #add batch to correct dataset group
                if (line_cnt < n_train_samples):
                    sine_train.append([current_batch_sine, len(sine_train)])
                    CAxis_train.append(current_batch_CAxis)
                else:
                    sine_test.append([current_batch_sine, len(sine_test)])
                    CAxis_test.append(current_batch_CAxis)

                #reset for next batch
                current_batch_sine = np.zeros((batch_size, nPoints), dtype = np.float32)
                current_batch_CAxis = np.zeros((batch_size, 3), dtype = np.float32)
                current_batch_sine[batch_cnt] = current_sine_arr
                current_batch_CAxis[batch_cnt] = current_triplet

            batch_cnt += 1

    f.close()

    return sine_train, CAxis_train, sine_test, CAxis_test
        

if __name__=='__main__':
    full_sine, full_CAxis = convert_CAxis_to_Sine_test("./datasets/", True)
