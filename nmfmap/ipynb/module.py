import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
import matplotlib
import time
import collections
import math
from PIL import Image

## Calculation of lineary matrix to convert y to Ay
## A depends on x (e.g. position of center of the beam or obliqutiy of planet or..)
def func_x(x, y):
    return A

## A (n_data, n_xd)

## Calculation of TSV
def TSV(mat):
    sum_tsv = 0
    Nx, Ny = np.shape(mat)
    
    # TSV terms from left to right 
    mat_2 = np.roll(mat, shift = 1, axis = 1) 
    sum_tsv += np.sum( (mat_2[:,1:Ny]-mat[:,1:Ny]) * (mat_2[:,1:Ny]-mat[:,1:Ny]) )
    
    # TSV terms from bottom to top  
    mat_3 = np.roll(mat, shift = 1, axis = 0) 
    sum_tsv += np.sum( (mat_3[1:Nx, :]-mat[1:Nx, :]) * (mat_3[1:Nx, :]-mat[1:Nx, :]) )
    
    #Return all TSV terms
    return sum_tsv

## Calculation of d_TSV
def d_TSV(mat):
    
    dif_c = 2*np.diff(mat,axis=1)
    dif_1 = np.pad(dif_c, [(0,0),(1,0)], mode = 'constant')
    dif_2 = np.pad(-dif_c, [(0,0),(0,1)], mode = 'constant')
    dif_c= 2*np.diff(mat,axis=0)
    dif_3 = np.pad(dif_c, [(1,0),(0,0)], mode = 'constant')
    dif_4 = np.pad(-dif_c, [(0,1),(0,0)], mode = 'constant')

    return dif_1 + dif_2 + dif_3 + dif_4
                            

def d_TSVold(mat):
    Nx, Ny = np.shape(mat)
    d_TSV_mat = np.zeros(np.shape(mat))
    mat_1 = np.roll(mat, shift = 1, axis = 1)
    mat_2 = np.roll(mat, shift = -1, axis = 1)
    mat_3 = np.roll(mat, shift = 1, axis = 0)
    mat_4 = np.roll(mat, shift = -1, axis = 0)
    dif_1 = 2 * (mat[:,1:Ny] - mat_1[:,1:Ny])
    dif_1 = np.pad(dif_1, [(0,0),(1,0)], mode = 'constant')
    dif_2 = 2 * (mat[:,0:Ny-1] - mat_2[:,0:Ny-1])
    dif_2 = np.pad(dif_2, [(0,0),(0,1)], mode = 'constant')
    dif_3 = 2 * (mat[1:Nx, :] - mat_3[1:Nx, :])
    dif_3 = np.pad(dif_3, [(1,0),(0,0)], mode = 'constant')
    dif_4 = 2 * (mat[0:Nx-1, :] - mat_4[0:Nx-1, :])
    dif_4 = np.pad(dif_4, [(0,1),(0,0)], mode = 'constant')

    return dif_1 + dif_2 + dif_3 + dif_4

## Cauculation of ||y-Ax||^2 + TSV

def F_TSV(data, A,  x_d, lambda_tsv):
    data_dif = data -  np.einsum("ijk,jk->i", A, x_d)
    return (np.dot(data_dif, data_dif)/2)  + TSV(x_d) *  lambda_tsv

def F_obs(data, A,  x_d):
    data_dif = data -  np.einsum("ijk,jk->i", A, x_d)
    return (np.dot(data_dif, data_dif)/2)

# Derivative of ||y-Ax||^2 + TSV (F_TSV)
##  np.dot(A.T, data_dif) is n_image vecgtor, d_TSV(x_d) is the n_image vecgtor or matrix
def dF_dx(data, A, x_d, lambda_tsv):
    data_dif = -(data - np.einsum("ijk,jk->i", A, x_d))
    return np.einsum("ijk,i->jk", A, data_dif) +lambda_tsv*  d_TSV(x_d)

## Calculation of Q(x, y) (or Q(P_L(y), y)) except for g(P_L(y))
## x_d2 = PL(y) (xvec1)
## x_d = y (xvec2)


### backtracking right term
def calc_Q_part(data, A,  x_d2, x_d, df_dx, L, lambda_tsv):
    Q_core = F_TSV(data, A, x_d, lambda_tsv) ## f(y) 
    Q_core += np.sum((x_d2 - x_d)*df_dx) + 0.5 * L * np.sum( (x_d2 - x_d) * (x_d2 - x_d))
    return Q_core

## Calculation of soft_thresholding (prox)
#   nx, ny = np.shape(x_d)
def soft_threshold_nonneg(x_d, eta):
    vec = np.zeros(np.shape(x_d))
    mask=x_d>eta
    vec[mask]=x_d[mask] - eta
    return vec

### For consistnecy with sparse modleing (Shiro Ikeda)
## box_flag=0 &&& cl_flag = 1
## nonnegative = 1


## Function for MFISTA
def mfista_func(I_init, d, A_ten, lambda_l1= 1e2, lambda_tsv= 1e-8, L_init= 1e4, eta=1.1, maxiter= 10000, max_iter2=100, 
                    miniter = 100, TD = 30, eps = 1e-5, print_func = False):

    ## Initialization
    mu, mu_new = 1, 1
    y = I_init
    x_prev = I_init
    cost_arr = []
    L = L_init
    
    ## The initial cost function
    cost_first = F_TSV(d, A_ten, I_init, lambda_tsv)
    cost_first += lambda_l1 * np.sum(np.abs(I_init))
    cost_temp, cost_prev = cost_first, cost_first

    ## Main Loop until iter_now < maxiter
    ## PL_(y) & y are updated in each iteration
    p1=0.
    p2=0.
    p3=0.
    for iter_now in range(maxiter):
        s1=time.time()
        cost_arr.append(cost_temp)
        
        ##df_dx(y)
        df_dx_now = dF_dx(d,A_ten, y, lambda_tsv) 
        
        ## Loop to estimate Lifshitz constant (L)
        ## L is the upper limit of df_dx_now
        s2=time.time()
        ## Backtracking
        for iter_now2 in range(max_iter2):
            
            y_now = soft_threshold_nonneg(y - (1/L) * df_dx_now, lambda_l1/L)
            Q_now = calc_Q_part(d, A_ten, y_now, y, df_dx_now, L,  lambda_tsv)
            F_now = F_TSV(d, A_ten, y_now,lambda_tsv)
            
            ## If y_now gives better value, break the loop
            if F_now <Q_now:
                break
            L = L*eta

        L = L/eta #Here we get Lifshitz constant
        s3=time.time()

        #Nesterov acceleration
        mu_new = (1+np.sqrt(1+4*mu*mu))/2
        F_now += lambda_l1 * np.sum(np.abs(y_now))
        if print_func:
            if iter_now % 50 == 0:
                print ("Current iteration: %d/%d,  L: %f, cost: %f, cost_chiquare:%f" % (iter_now, maxiter, L, cost_temp, F_obs(d, A_ten, y_now)))

        ## Updating y & x_k
        if F_now < cost_prev:
            cost_temp = F_now
            tmpa = (1-mu)/mu_new
            x_k = soft_threshold_nonneg(y - (1/L) * df_dx_now, lambda_l1/L)
            y = x_k + ((mu-1)/mu_new) * (x_k - x_prev) 
            x_prev = x_k
            
        else:
            cost_temp = F_now
            tmpa = 1-(mu/mu_new)
            tmpa2 =(mu/mu_new)
            x_k = soft_threshold_nonneg(y - (1/L) * df_dx_now, lambda_l1/L)
            y = tmpa2 * x_k + tmpa * x_prev       
            x_prev = x_k
            
        if(iter_now>miniter) and np.abs(cost_arr[iter_now-TD]-cost_arr[iter_now])<cost_arr[iter_now]*eps:
            break

        mu = mu_new
        s4=time.time()
        p1+=s2-s1
        p2+=s3-s2
        p3+=s4-s3
    print(p1,p2,p3,"SEC in total")
    return y




## Making data for che-mapping
class random_generator:
    def __init__(self, N_data, Nx, Ny):
        self.N_data = N_data
        self.Nx = Nx
        self.Ny = Ny
    def make_data(self, I_true, width_max):

        d=[]
        g=[]

        deltaa = 2+np.random.rand(self.N_data)*width_max
        deltaa = deltaa.astype(np.int64) #width of beams       
        ixrand = np.random.rand(self.N_data)*self.Nx
        ixrand = ixrand.astype(np.int64) # x-position of beams       
        iyrand = np.random.rand(self.N_data)*self.Ny  # y-position of beams
        iyrand = iyrand.astype(np.int64) 

        for i in range(self.N_data):
            ix=ixrand[i]
            iy=iyrand[i]
            delta=deltaa[i]
            ix1=max(ix-delta,0)
            ix2=min(self.Nx,ix+delta)
            iy1=max(iy-delta,0)
            iy2=min(self.Ny,iy+delta)
            val=np.sum(I_true[ix1:ix2,iy1:iy2])
            d.append(val)
            gzero=np.zeros((self.Nx,self.Ny),dtype="float")
            gzero[ix1:ix2,iy1:iy2]=1.0
            g.append(gzero)
        return np.array(d), np.array(g)

    def make_colordata(self, I_true, width_max):

        d=[]
        g=[]

        deltaa = 2+np.random.rand(self.N_data)*width_max
        deltaa = deltaa.astype(np.int64) #width of beams       
        ixrand = np.random.rand(self.N_data)*self.Nx
        ixrand = ixrand.astype(np.int64) # x-position of beams       
        iyrand = np.random.rand(self.N_data)*self.Ny  # y-position of beams
        iyrand = iyrand.astype(np.int64) 

        for i in range(self.N_data):
            ix=ixrand[i]
            iy=iyrand[i]
            delta=deltaa[i]
            ix1=max(ix-delta,0)
            ix2=min(self.Nx,ix+delta)
            iy1=max(iy-delta,0)
            iy2=min(self.Ny,iy+delta)
            val=np.sum(I_true[ix1:ix2,iy1:iy2,:],axis=(0,1))
            d.append(val)
            gzero=np.zeros((self.Nx,self.Ny),dtype="float")
            gzero[ix1:ix2,iy1:iy2]=1.0
            g.append(gzero)
        return np.array(d), np.array(g)
    

### 
class CVPlotter(object):
    def __init__(self, nv, nh, L1_list, Ltsv_list):
        self.nh = nh
        self.nv = nv

        self.left_margin = 0.1
        self.right_margin = 0.1
        self.bottom_margin = 0.1
        self.top_margin = 0.1
        total_width = 1.0 - (self.left_margin + self.right_margin)
        total_height = 1.0 - (self.bottom_margin + self.top_margin)
        dx = total_width / float(self.nh)
        dy = total_height / float(self.nv)
        self.dx = min(dx, dy)
        self.dy = self.dx
        f = plt.figure(num='CVPlot', figsize=(10,10))
        plt.clf()
        left = self.left_margin
        bottom = self.bottom_margin
        height = self.dy * self.nv
        width = self.dx * self.nh
        outer_frame = plt.axes([left, bottom, width, height])
        outer_frame.set_xlim(-0.5, self.nh - 0.5)
        outer_frame.set_ylim(-0.5, self.nv - 0.5)
        outer_frame.set_xlabel('log$_{10}$($\\Lambda_{t}$)', fontsize = 26)
        outer_frame.set_ylabel('log$_{10}$($\\Lambda_{l}$)', fontsize = 26)
        outer_frame.tick_params(labelsize = 22)
        outer_frame.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nh))))
        outer_frame.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(list(range(self.nv))))
        outer_frame.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(Ltsv_list[int(x)]))))
        outer_frame.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: int(math.log10(L1_list[int(x)]))))
        
        self.L1_list = L1_list
        self.Ltsv_list = Ltsv_list
        
        self.axes_list = collections.defaultdict(dict)

    def make_cv_figure(self, folder_name, outfile, file_name_head=""):

        for j in range(self.nv):
            for i in range(self.nh-1,-1,-1):
                data_file_name = "%s/%sl1_%d_ltsv_%d.npy" % (folder_name, file_name_head, np.log10(self.L1_list[i]), np.log10(self.Ltsv_list[j]))
                self.plotimage(i, j, np.load(data_file_name))

        self.draw()
        self.savefig(outfile)
        plt.show()
        plt.close()


    def plotimage(self, row, column, data):
        left = self.left_margin + column * self.dx
        bottom = self.bottom_margin + row * self.dy
        height = self.dx
        width = self.dy
        nx, ny = data.shape
        a = plt.axes([left, bottom, width, height])

        a.imshow(data,cmap="gray")
        a.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        a.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        (x_cen, y_cen) =  (np.unravel_index(np.argmax(data), data.shape))
        (x_cen, y_cen) =  (np.unravel_index(np.argmax(data), data.shape))
#        a.set_xlim(121+width,121-width)
#        a.set_ylim(y_cen-width,y_cen+width)
        self.axes_list[row][column] = a

    def draw(self):
        plt.draw()



    def savefig(self, figfile):
        plt.savefig(figfile,  dpi=300)




## Main class for imaging with sparse modeling
class main_sparse:

    def __init__(self, d, A_ten, eta = 1.1, maxiter= 10000, maxiter2=100, miniter = 100, TD = 50, eps = 1e-5, print_func = False):
        self.maxiter = maxiter
        self.maxiter2 = maxiter2
        self.miniter = miniter
        self.TD = TD
        self.eps = eps
        self.print_func = print_func
        self.A = A_ten
        self.d = d
        self.eta = eta
        self.N_data, self.Nx, self.Ny = np.shape(A_ten)
        
    def cost_evaluate(self, I_now, lambda_l1, lambda_tsv):

        return F_obs(self.d, self.A, I_now), lambda_tsv * TSV(I_now), lambda_l1 * np.sum(np.abs(I_now))
    

    def make_random_index(self, n_fold):

        random_index = np.arange(len(self.d), dtype = np.int64)
        np.random.shuffle(random_index)
        random_index_mod = random_index % n_fold
        return (random_index_mod)

    def save_data(self, arr, folder_name, file_name):

        np.save("%s/%s" % (folder_name, file_name), arr)


    def cv(self, lambda_l1_row, lambda_tsv_row, n_fold, file_name_head = "cv", folder_name = "./", file_out = "cv_result"):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)


        folder_d_name=folder_name + "_data"
        if not os.path.isdir(folder_d_name):
            os.mkdir(folder_d_name)

        self.MSE_result = []
        self.MSE_sigma_result = []
        self.l1_result = []
        self.ltsv_result= []


        for l1_now in lambda_l1_row:
            for ltsv_now in lambda_tsv_row:
                rand_index = self.make_random_index(n_fold)
                MSE = []

                for i in range(n_fold):

                    if i==0:
                        I_init = np.ones((self.Nx, self.Ny))
                    else:
                        I_init = I_est

                    ## Separating the data into taining data and test data
                    A_ten_for_est = self.A[rand_index != i]
                    d_for_est = self.d[rand_index != i]
                    A_ten_for_test = self.A[rand_index == i]
                    d_for_test = self.d[rand_index == i]
                    t1 = time.time() 
                    I_est = mfista_func(I_init, d_for_est , A_ten = A_ten_for_est, print_func = self.print_func, eta = self.eta, 
                        lambda_tsv = ltsv_now,lambda_l1= l1_now, maxiter = self.maxiter,miniter = 35, TD = 30)
                    t2 = time.time() 
                    MSE_now = F_obs(d_for_test, A_ten_for_test, I_est)/(len(d_for_test)*1.0)
                    MSE.append(MSE_now)
                    print("%d/%d, l1: %d/%d, ltsv: %d/%d. elapsed time: %f" % (i, n_fold, np.log10(l1_now), np.log10(np.max(lambda_l1_row)), 
                        np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), t2-t1))


                self.MSE_result.append(np.mean(MSE))
                self.MSE_sigma_result.append(np.std(MSE))
                self.l1_result.append(l1_now)
                self.ltsv_result.append(ltsv_now)

                I_est = self.solve_and_make_figure(l1_now, ltsv_now, file_name ="%s/%s_l1_%d_ltsv_%d.png" % (folder_name, file_name_head, np.log10(l1_now), np.log10(ltsv_now)))
                self.save_data(I_est, folder_d_name, "%s_l1_%d_ltsv_%d" % (file_name_head, np.log10(l1_now), np.log10(ltsv_now)))



        # Outputting result
        self.cv_result_out(file_out + ".dat")
        plotter = CVPlotter(len(l1_arr), len(ltsv_arr), l1_arr, ltsv_arr)
        plotter.make_cv_figure(folder_name, outfile=file_out+".png", file_name_head=file_name_head)


    def cv_result_out(self, file_out):

        file_now = open(file_out, "w")
        print("MSE, MSE_std, l1, ltsv", file = file_now)

        for i in range(len(self.MSE_result)):
            print ("%f, %f, %f, %f" % (self.MSE_result[i], self.MSE_sigma_result[i],self.l1_result[i], self.ltsv_result[i]), file = file_now)
        file_now.close()



    def solve_without_cv(self, lambda_l1_row, lambda_tsv_row, L_init,file_name_head = "cv", folder_name = "./", print_status = True, file_out = "cv_result"):
        
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        folder_d_name=folder_name + "_data"

        if not os.path.isdir(folder_d_name):
            os.mkdir(folder_d_name)
        for l1_now in lambda_l1_row:
            for ltsv_now in lambda_tsv_row:
                t1 = time.time() 


                I_est = self.solve_and_make_figure(l1_now, ltsv_now, file_name ="%s/%s_l1_%d_ltsv_%d.png" % (folder_name, file_name_head, np.log10(l1_now), np.log10(ltsv_now)))
                t2 = time.time()

                self.save_data(I_est, folder_d_name, "l1_%d_ltsv_%d" % (np.log10(l1_now), np.log10(ltsv_now)))

                if print_status:
                    chi_term, tsv_term, l1_term =  self.cost_evaluate(I_est, l1_now, ltsv_now)
                    print("l1: %.1f/%.1f, ltsv: %.1f/%.1f. elapsed time: %e" % (np.log10(l1_now), np.log10(np.max(lambda_l1_row)), 
                        np.log10(ltsv_now), np.log10(np.max(lambda_tsv_row)), t2-t1))
                    print ("total cost:%e, chi:%e, l1:%e, ltsv:%e \n" % (chi_term+tsv_term + l1_term, chi_term, l1_term, tsv_term))

        plotter = CVPlotter(len(lambda_l1_row), len(lambda_tsv_row), lambda_l1_row, lambda_tsv_row)
        plotter.make_cv_figure(folder_name, outfile =file_out + ".png", file_name_head=file_name_head)




    def solve_and_make_figure(self, lambda_l1, lambda_tsv, file_name, original_file =None):

        I_init = np.ones((self.Nx, self.Ny))
        I_est = mfista_func(I_init, self.d , A_ten =self.A, print_func = self.print_func, eta = self.eta, 
                        lambda_tsv =lambda_tsv,lambda_l1= lambda_l1, maxiter = self.maxiter)

        if original_file is None:
            fig =plt.figure()
            plt.imshow(I_est, cmap='gray')
            plt.savefig(file_name, dpi = 200)
            plt.close()

        else:
            fig =plt.figure()
            ax=fig.add_subplot(121)
            ax.imshow(I_est, cmap='gray')
            ax=fig.add_subplot(122)
            ax.imshow(img, cmap='gray')
            plt.savefig(file_name, dpi = 200)
            plt.close()
        return I_est
