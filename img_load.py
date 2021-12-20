import numpy as np
import matplotlib.pyplot as plt

def readpara(filepath,alltabs=False):
    import csv

    with open(filepath+'parameters.txt') as csvfile:
        reader=csv.DictReader(csvfile,delimiter='\t')
        fieldnames=reader.fieldnames
        if alltabs:
            result=[]
            for row in reader:
                para=[]
                for fn in fieldnames:
                    para.append(float(row[fn]))

                for value in row[None]:
                    para.append(float(value))
            
                result.append(para)
            return result
        else:
            result=[]
            filenum=[]
            for row in reader:
                result.append(float(row[fieldnames[1]]))
                filenum.append(int(row[fieldnames[0]]))
            return result, filenum

def readdata(filepath,alltabs=False):
    import csv

    with open(filepath) as csvfile:
        reader=csv.DictReader(csvfile,delimiter='\t')
        fieldnames=reader.fieldnames
        
        result=[]
        for row in reader:
            para=[]
            for fn in fieldnames:
                para.append(float(row[fn]))
            for value in row[None]:
                para.append(float(value))
           
            result.append(para)
        return result, fieldnames
        
def readindex(filepath):
    import csv

    with open(filepath+'index.txt') as csvfile:
        reader=csv.reader(csvfile,delimiter='\t')

        result=[]
        fldnum=[]
        for row in reader:
            result.append(row[1])
            fldnum.append(row[0])
        return result,fldnum
        
def read_binary(fname,raw=False, signedRaw=False):
        
    import os
    
    with open(fname, "rb") as f:
        dim=[int.from_bytes(f.read(4), byteorder='big'),int.from_bytes(f.read(4), byteorder='big')]
        data_size=int((os.stat(fname).st_size-8)/dim[0]/dim[1])

        if raw:
            ODimg = np.zeros(dim,dtype=int)
            for i in range(dim[0]):
                for j in range(dim[1]):
                    ODimg[i][j] = int.from_bytes(f.read(data_size), byteorder='big', signed=signedRaw)
        else:
            ODimg = np.zeros(dim,dtype=float)
            for i in range(dim[0]):
                for j in range(dim[1]):
                    ODimg[i][j] = int.from_bytes(f.read(data_size), byteorder='big', signed=True)/10000.
                    
    return ODimg, dim
        
def load_bimg(filepath=None,para=None,raw=False, imgnum=[]):

    from pathlib import Path
    import ntpath

    #create filelist
    filelist=[]
    if filepath==None:      

        #pending, should find a way to manually choose the file
        #filelist = DIALOG_PICKFILE(/MULTIPLE_FILES,/READ,PATH= SCOPE_VARFETCH('workingdirectory', LEVEL=1, /ENTER),get_path=filepath)
        #(SCOPE_VARFETCH('workingdirectory', /ENTER, LEVEL=1)) = filepath

        parameters, filenum=readpara(filepath)

        if para!=None:
            ind=np.where( np.array(parameters) == para)[0]
            numfiles=len(ind)
            for i in ind:
                filelist.append(filepath+string(filenum[i]))
        else:
            numfiles=len(filelist)
    else:
        parameters, filenum=readpara(filepath)

        if para!=None:
            ind=np.where( np.array(parameters) == para)[0]
            numfiles=len(ind)
            for i in ind:
                filelist.append(filepath+str(filenum[i]))
        else:
            numfiles=len(parameters)
            for i in range(numfiles):
                filelist.append(filepath+str(filenum[i]))

    dataDim=[1]
            
    if np.asarray(imgnum).size!=0:
        filelist = [filelist[inum] for inum in imgnum]
    
    #load binary images
    for fname in filelist:
        if dataDim[0]==1:
            od=[]
            raw1=[]
            raw2=[]
            ramp_para=[]
        
        if raw==False:
            ODimg, dataDim = read_binary(fname);              
            od.append(ODimg)
                
        #load raw images
        else:
            imgpath=filepath+"rawimg_"+ntpath.basename(fname)                
            if Path(imgpath).exists():
                ODimg, dataDim = read_binary(imgpath,raw=True)
                raw1.append(ODimg[0:dataDim[0]//2-1,:])
                raw2.append(ODimg[dataDim[0]//2:dataDim[0]-1,:])
            else:
                print("File path does not exist.")
                
#         ramp_para.append(parameters[np.where( np.array(filenum) == int(ntpath.basename(fname)))[0][0]])
        ramp_para.append(0)
    
    if np.size(filelist)>0:
        od=np.asarray(od,dtype=np.float32)
        raw1=np.asarray(raw1,dtype=np.float32)
        raw2=np.asarray(raw2,dtype=np.float32)
    
        print('load data from ' + filepath)
        print('od size', od.shape)
                       
        return {'od':od,'raw1':raw1,'raw2':raw2,'para':ramp_para}
    else: 
        return {'od':[],'raw1':[],'raw2':[],'para':[]}


def imgplay(img,figtitle='Frame Number: ',delay=300):
    import matplotlib.animation as animation
    framenum=np.shape(img)[0]
    fig=plt.figure()
    fig.suptitle(figtitle + str(0), fontsize=12)
    im=plt.imshow(img[0,:,:],animated=True,interpolation='nearest')#,extent=[0*0.85,15*0.85,25*0.85,0*0.85])
    plt.colorbar()
    #plt.show()
    ani = animation.FuncAnimation(fig, updatefig ,frames=range(framenum), interval=delay, blit=True,repeat=False)


def updatefig(frame):
    im.set_data(img[frame,:,:])
    fig.suptitle(figtitle + str(frame), fontsize=12)
    #im1.set_data(result['od'][frame,10:55,74:81])
    return im,


