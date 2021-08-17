import numpy as np
import sklearn.cluster 
import sklearn.metrics
import hmmlearn.hmm
import PIL.Image
import pickle as pkl

class Tools:

    def gilbert2d(width, height):

        if width >= height:
            yield from Tools.generate2d(0, 0, width, 0, 0, height)
        else:
            yield from Tools.generate2d(0, 0, 0, height, width, 0)


    def sgn(x):
        return -1 if x < 0 else (1 if x > 0 else 0)


    def generate2d(x, y, ax, ay, bx, by):

        w = abs(ax + ay)
        h = abs(bx + by)

        (dax, day) = (Tools.sgn(ax), Tools.sgn(ay)) # unit major direction
        (dbx, dby) = (Tools.sgn(bx), Tools.sgn(by)) # unit orthogonal direction

        if h == 1:
            # trivial row fill
            for i in range(0, w):
                yield(x, y)
                (x, y) = (x + dax, y + day)
            return

        if w == 1:
            # trivial column fill
            for i in range(0, h):
                yield(x, y)
                (x, y) = (x + dbx, y + dby)
            return

        (ax2, ay2) = (ax//2, ay//2)
        (bx2, by2) = (bx//2, by//2)

        w2 = abs(ax2 + ay2)
        h2 = abs(bx2 + by2)

        if 2*w > 3*h:
            if (w2 % 2) and (w > 2):
                # prefer even steps
                (ax2, ay2) = (ax2 + dax, ay2 + day)

            # long case: split in two parts only
            yield from Tools.generate2d(x, y, ax2, ay2, bx, by)
            yield from Tools.generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

        else:
            if (h2 % 2) and (h > 2):
                # prefer even steps
                (bx2, by2) = (bx2 + dbx, by2 + dby)

            # standard case: one step up, one long horizontal, one step down
            yield from Tools.generate2d(x, y, bx2, by2, ax2, ay2)
            yield from Tools.generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
            yield from Tools.generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))
        
    def path_HP(height,width):
        l=[]
        for (x,y) in Tools.gilbert2d(height,width):
            l.append((x,y))
        return np.array(l)

    def array_to_chain(tab,path_curve):
        res=np.zeros((len(path_curve)))
        i=0
        while i<len(path_curve):
            res[i]=tab[path_curve[i,0],path_curve[i,1]]
            i+=1
        return res

    def chain_to_array(tab,height,width,path_curve):
        res=np.zeros((height,width))
        for idx, coords in enumerate(path_curve):
            res[coords[0],coords[1]]=tab[idx]
        return res

class Image:

    def __init__(self,im_path):
        self.im=np.array(PIL.Image.open(im_path).convert('L'))
        self.components=None
        self.im_cleaned=None
        self.h=None
        self.w=None
        self.path_curve=None
        self.hmm_model=None

    def set_dim(self):
        self.h=self.im.shape[0]
        self.w=self.im.shape[1]
    
    def set_path_curve(self):
        self.path_curve=Tools.path_HP(self.h,self.w)
    
    def setting(self):
        self.set_dim()
        self.set_path_curve()

    def get_components(self):
        return self.components

    def transform_1D(self):
        self.im=Tools.array_to_chain(self.im,self.path_curve)
    
    def transform_2D(self):
        self.im=Tools.chain_to_array(self.im,self.h,self.w,self.path_curve)
    
    def set_components(self):
        n_components=range(2,100)
        sample_for_kmeans=np.random.choice(self.im,size=self.im.shape[0]//100,replace=False).reshape(-1,1)
        silhouettes=np.zeros(10)
        for n in n_components:
            kmeans=sklearn.cluster.KMeans(n_clusters=n)
            kmeans.fit(sample_for_kmeans)
            silhouettes[n-10]=sklearn.metrics.silhouette_score(sample_for_kmeans,kmeans.labels_)
        self.components=np.argmax(silhouettes)+10

    def get_im_cleaned(self):
        return self.im_cleaned

    def set_hmm_model(self):
        if self.components!=None:
            model=hmmlearn.hmm.GaussianHMM(n_components=self.components,covariance_type='diag',params='stmc')
            data=self.im.reshape(-1,1)
            model.fit(data)
            self.hmm_model=model 
    
    def clean_image(self,border,display=False,save=False,dir_path=''):
        if self.hmm_model!=None:
            means=self.hmm_model.means_
            means_sorted=sorted(means)
            hidden_states=self.hmm_model.predict(self.im.reshape(-1,1)) #using Viterbi algorithm
            cleaned_1D_array=np.where(means[hidden_states]<=means_sorted[border],0,255)
            self.im_cleaned=Tools.chain_to_array(cleaned_1D_array.reshape(len(self.path_curve)),self.h,self.w,self.path_curve)
            if display:
                PIL.Image.fromarray(self.im_cleaned).show()
            if save:
                if len(dir_path)==0:
                    print('Please enter the directory path to save the cleared image')
                else:
                    PIL.Image.fromarray(self.im_cleaned).convert('RGB').save(dir_path+'image_clear#'+str(border)+'.jpg')
    
    def save_hmm_model(self,path):
        with open(path,'wb') as f:
            pkl.dump(self.hmm_model,f)
            f.close()
    
    def load_hmm_model(self,path):
        with open(path,'rb') as f:
            self.hmm_model=pkl.load(f)
            f.close()
