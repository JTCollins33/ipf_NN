import numpy as np
import math
import torchvision.transforms as transforms

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

def find_CAxis_Stats(f1, f2, f3):
    amp = int(100*(1-f3))
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
        rad = ((4*math.pi)/(1.0*nPoints))*i
        # sine_arr[i-1] = (amp*math.sin(rad)+v_shift)/(1.0*(amp+v_shift)) * 255
        sine_arr[i-1] = ((amp*math.sin(rad)+v_shift)/(1.0*(amp+v_shift))-0.5)*2

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
            sine_transform = get_transform()
            sine_arr = sine_transform(sine_arr)
        full_list.append(sine_arr)
        full_CAxis.append(CAxis_arr2)
    return full_list, full_CAxis
        

if __name__=='__main__':
    full_sine, full_CAxis = convert_CAxis_to_Sine("./datasets/", False)