import budformer as et
import torch
import clustering_main as cm

#params = et.ParameterProvider("series-weather.config")

device_id = torch.cuda.current_device()

n_reps = 3

cm.run(config="series-weather.config",set="dane_pogodowe.csv",dump_location="/net/afscra/people/plgmftrzcinski/clustering_outputs",output="output",maxclusters=10,window_length=720,prediction_window=30,max_clustering_samples=1000, split_date=True, drop_id=False)
ret_list = []


for i in range(n_reps):

    #RETURNS RAW DATA!!!
    inp, out, pred = cm.run(config="series-energy.config",set="energydata_cleared.csv",dump_location=".",output="energy",drop_id=False, split_date=True, maxclusters=10,window_length=720,prediction_window=30,max_clustering_samples=100)
    file_ret = open("return_energy.txt","a")
    file_ret.write(str((inp,out,pred)))
    file_ret.write("\n\n")
    file_ret.close()
    