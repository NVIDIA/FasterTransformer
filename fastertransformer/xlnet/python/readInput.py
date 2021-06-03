import numpy as np


if __name__=="__main__":
    data = np.load('./data.npz')
    batch_size = 8
    max_seq_length=128

    arr_input_ids=data["input_ids:0"];
    arr_input_mask=data["input_mask:0"];
    arr_segment_ids=data["segment_ids:0"];
    arr_label_ids=data["label_ids:0"];

    print("Length {} {} {} {} ".format(arr_input_ids.shape, arr_input_mask.shape, arr_segment_ids.shape,
            arr_label_ids.shape))
    for i in range(arr_input_ids.size/128):
        if i % 500 == 0:
            input_ids=arr_input_ids[i,:]
            print("Writing example {} in shape {} with {} ".format(i,input_ids.shape, input_ids))


