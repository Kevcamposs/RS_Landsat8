# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:05:16 2022

@author: Kevin
"""
class Basic_L8:
    def __init__(self, img, normalize = True, float_ = True, normal_minmax=True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar = rio.open(self.img).read(1)
        self.crs = rio.open(self.img).transform
        self.profile = rio.open(self.img).profile
        
        if float_==True:
            self.img_ar.astype('f4')
            self.profile['dtype'] = 'float64'
        
        if normalize == True:
            if normal_minmax == True:
                a_min = np.min(self.img_ar)
                a_max = np.max(self.img_ar)
                self.img_ar = (self.img_ar - a_min)/(a_min - a_max)                
            elif normal_minmax == False:
                a_2 = np.nanpercentile(self.img_ar, 2)
                a_98 = np.nanpercentile(self.img_ar, 98)
                self.img_ar = (self.img_ar - a_2)/(a_98 - a_2)
        
        return print('Carga de imágenes, completado.')
    ###########################################################################
    def correction(self, metadata, band, output = 'output.tif', plot = 'ps'):
        file = open(metadata, 'r', encoding='utf-8')
        
        # Extracción de datos de la metadata
        ML_index = range(165,176); ML = []; AL_index = range(176,187); AL = []
        MaxL_index = range(97, 119, 2); MaxL = []; MinL_index = range(98, 119, 2); MinL = []
        Mp_index = range(186,196); Mp = []; Ap_index = range(196,205); Ap = []
        Maxp_index = range(121, 139, 2); Maxp = []; Minp_index = range(122, 139, 2); Minp = []
        K1_index = range(207, 210, 2); K1 = []; K2_index = range(208, 211, 2); K2 = []
        Thse_index = 76; Date_index = 4
        
        for position, line in enumerate(file):
            if position in ML_index:
                ML.append(float(str(line[-12:-1].replace('=', ''))))   # Multiplicativo radiancia
            elif position in AL_index:
                AL.append(float(str(line[-10:-1].replace('=', ''))))   # Aditivo radiancia
            elif position in MaxL_index:
                MaxL.append(float(str(line[-10:-1].replace('=', '')))) # Radiancia máxima
            elif position in MinL_index:
                MinL.append(float(str(line[-10:-1].replace('=', '')))) # Radiancia mínima
            elif position in Mp_index:
                Mp.append(float(str(line[-12:-1].replace('=', ''))))   # Multiplicativo reflectancia
            elif position in Ap_index:
                Ap.append(float(str(line[-10:-1].replace('=', ''))))   # Aditivo reflectancia
            elif position in Maxp_index:
                Maxp.append(float(str(line[-10:-1].replace('=', '')))) # Reflectancia máxima
            elif position in Minp_index:
                Minp.append(float(str(line[-10:-1].replace('=', '')))) # Reflectancia 
            elif position in K1_index:
                K1.append(float(str(line[-12:-1].replace('=', ''))))   # Constante K1
            elif position in K2_index:
                K2.append(float(str(line[-12:-1].replace('=', ''))))   # Constante K2
            elif position == Thse_index:
                Thse = float(str(line[-12:-1]))                        # Elevación solar
            elif position == Date_index:
                Date = int(str(line[-10:-7]))                          # Fecha juliana
        file.close()
        
        ########## Corrección atmosférica
        ### 1. Calibración radiométrica: ND -> radiancia
        ### 2. Cálculo de reflectancia: radiancia -> reflectancia aparente
        
        from numpy import amin, pi, cos, sin
        self.LA = self.img_ar*ML[band-1] + AL[band-1]      # (1) Radiancia espectral del sensor
        
        self.pA = (self.img_ar*Mp[band-3] + Ap[band-3])/sin(Thse*pi/180) # (2.1) Reflectancia TOA
        
        d = 1 - 0.0167*cos((Date-3)*2*pi/365)            # Distancia Tierra-Sol
        Lmin = amin(self.img_ar)*ML[band-1] + AL[band-1] # Radiancia del menor número digital
        ESUNA = pi*d*MaxL[band-1]/Maxp[band-1]           # Irradiancia Media Solar exo-atmosférica
        LDOS1 = 0.01*ESUNA*sin(Thse*pi/180)/(pi*d**2)    # Radiancia del objeto oscuro
        Lp = Lmin - LDOS1                                # Path radiance. Efecto bruma
        
        self.ps = pi*(self.LA - Lp)*d**2/(ESUNA*sin(Thse*pi/180)) # (2.2) Reflectancia en superficie
        
        if plot == 'ps':
            self.img_ar = self.ps
            
        import rasterio

        with rasterio.open(output, 'w', **self.profile) as dst:
            dst.write(self.img_ar, indexes=1)

        return print('Proceso finalizado.')
        
    ###########################################################################
    def plot(self, title = 'Landsat 8', transform = True, cmap = 'gray'):
        from rasterio.plot import show
        if transform == True:
            show(self.img_ar, transform = self.crs, title = title, cmap = cmap)
        else:
            show(self.img_ar, title = title, cmap = cmap)
            
    def plot_bT(self, title = 'Landsat 8', transform = True, cmap = 'gray'):
        from rasterio.plot import show
        if transform == True:
            show(self.bT, transform = self.crs, title = title, cmap = cmap)
        else:
            show(self.bT, title = title, cmap = cmap)
    ###########################################################################
    def clip_shp(self, shp, output):
        from osgeo import gdal
        import numpy as np
        gdal.Warp(output, self.img, cutlineDSName = shp,cropToCutline = True, dstNodata= np.nan)
        return print('Proceso finalizado.')
    ###########################################################################  
    def clip_points(self, points, output):
        from osgeo import gdal
        gdal.Translate(output, self.img, projWin = points)
        return print('Proceso finalizado.')

###############################################################################
class Composite_L8:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1)
        self.img_ar2 = rio.open(self.img[1]).read(1)
        self.img_ar3 = rio.open(self.img[2]).read(1)
        
        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')
            self.img_ar3.astype('f4')
            
        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)
            self.img_ar3 = Normalize(self.img_ar3)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def composite(self):
        import numpy as np
        self.com = np.stack((self.img_ar1, self.img_ar2, self.img_ar3))
        
        return print('Composite completado.')
    ###########################################################################
    def plot(self, title = 'Landsat 8 - composite', save = True, 
             output = 'com.TIF', shp = None):
        import matplotlib.pyplot as plt, geopandas as gpd
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        from rasterio.plot import show
        show(self.com, transform = self.crs, title= title, ax = ax)
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = 'black', markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)
        plt.tight_layout()
        
###############################################################################
class ND_Index:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1)
        self.img_ar2 = rio.open(self.img[1]).read(1)

        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')
        
        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def Diff(self):
        import numpy as np
        self.index = np.where((self.img_ar1 + self.img_ar2) == 0, # Si se cumple esto...
                              0,   # asignar el siguiente valor, en todo caso...
                              (self.img_ar1 - self.img_ar2)/(self.img_ar1 + self.img_ar2))
        return print('Cálculo de índice completo.')

    ###########################################################################
    def plot(self, title = 'Landsat 8 - Index', save = True, 
             output = 'diff_index.TIF', shp = None, cmap = 'gray', cmap_inver = False):
        import matplotlib.pyplot as plt, geopandas as gpd
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        
        if cmap_inver == True:
            cmap = plt.cm.get_cmap(cmap).reversed()
        # Barra de color
        img = ax.imshow(self.index, cmap = cmap, vmin = -1, vmax = 1)
        fig.colorbar(img, ax = ax)
        
        show(self.index, transform = self.crs, title= title, ax = ax, cmap = cmap,
             vmin = -1, vmax = 1)
        
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = 'black', markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)
   
###############################################################################
class VARI_Index:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1) # B2: Blue
        self.img_ar2 = rio.open(self.img[1]).read(1) # B3: Green
        self.img_ar3 = rio.open(self.img[2]).read(1) # B4: Red

        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')
            self.img_ar3.astype('f4')
        
        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)
            self.img_ar3 = Normalize(self.img_ar3)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def VARI(self):
        import numpy as np
        self.index = np.where((self.img_ar2 + self.img_ar3 - self.img_ar1) == 0,
                              0,
                              (self.img_ar2 - self.img_ar3)/(self.img_ar2+self.img_ar3-self.img_ar1))
        return print('Cálculo de índice completo.')

    ###########################################################################
    def plot(self, title = 'Landsat 8 - Index', save = True, 
             output = 'diff_index.TIF', shp = None, cmap = 'gray', cmap_inver = False):
        import matplotlib.pyplot as plt, geopandas as gpd
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        
        if cmap_inver == True:
            cmap = plt.cm.get_cmap(cmap).reversed()
        
        img = ax.imshow(self.index, cmap = cmap)
        fig.colorbar(img, ax = ax)
        
        show(self.index, transform = self.crs, title= title, ax = ax, cmap = cmap)
        
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = cmap, markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)
   
###############################################################################
class ARVI_Index:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1) # B2: Blue
        self.img_ar2 = rio.open(self.img[1]).read(1) # B4: Red
        self.img_ar3 = rio.open(self.img[2]).read(1) # B5: NIR

        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')
            self.img_ar3.astype('f4')
        
        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)
            self.img_ar3 = Normalize(self.img_ar3)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def ARVI(self):
        import numpy as np
        self.index = np.where((self.img_ar3 + 2*self.img_ar2 + self.img_ar1) == 0,
                              0,
                              (self.img_ar3-2*self.img_ar2+self.img_ar1)/(self.img_ar3+2*self.img_ar2+self.img_ar1))
        return print('Cálculo de índice completo.')

    ###########################################################################
    def plot(self, title = 'Landsat 8 - Index', save = True, 
             output = 'diff_index.TIF', shp = None, cmap = 'gray', cmap_inver = False):
        import matplotlib.pyplot as plt, geopandas as gpd
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        
        if cmap_inver == True:
            cmap = plt.cm.get_cmap(cmap).reversed()
        
        img = ax.imshow(self.index, cmap = cmap, vmin = -1, vmax = 1)
        fig.colorbar(img, ax = ax)
        
        show(self.index, transform = self.crs, title= title, ax = ax, cmap = cmap)
        
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = cmap, markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)   
            
###############################################################################
class AVI_Index:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1) # B4: Red
        self.img_ar2 = rio.open(self.img[1]).read(1) # B5: NIR

        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')

        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def AVI(self):
        self.index = (self.img_ar2*(1-self.img_ar1)*(self.img_ar2-self.img_ar1))**1/3
        
        return print('Cálculo de índice completo.')

    ###########################################################################
    def plot(self, title = 'Landsat 8 - Index', save = True, 
             output = 'diff_index.TIF', shp = None, cmap = 'gray', cmap_inver = False):
        import matplotlib.pyplot as plt, geopandas as gpd
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        
        if cmap_inver == True:
            cmap = plt.cm.get_cmap(cmap).reversed()
        
        img = ax.imshow(self.index, cmap = cmap)
        fig.colorbar(img, ax = ax)
        
        show(self.index, transform = self.crs, title= title, ax = ax, cmap = cmap)
        
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = cmap, markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)    

###############################################################################
class BI_Index:
    def __init__(self, img, normalize = True, float_ = True):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar1 = rio.open(self.img[0]).read(1) # B2: Blue
        self.img_ar2 = rio.open(self.img[1]).read(1) # B3: Green
        self.img_ar3 = rio.open(self.img[2]).read(1) # B4: Red

        self.crs = rio.open(self.img[0]).transform
        
        if float_== True:
            self.img_ar1.astype('f4'); self.img_ar2.astype('f4')
            self.img_ar3.astype('f4')
        
        if normalize == True:
            def Normalize(a): 
                a_min = np.min(a); a_max = np.max(a)
                return (a - a_min)/(a_max - a_min)
            
            self.img_ar1 = Normalize(self.img_ar1)
            self.img_ar2 = Normalize(self.img_ar2)
            self.img_ar3 = Normalize(self.img_ar3)

        return print('Las imágenes se cargaron correctamente.')
    ###########################################################################
    def BI(self):
        import numpy as np
        self.index = np.where((self.img_ar1 + self.img_ar3 + self.img_ar2) == 0,
                              0,
                              (self.img_ar1+self.img_ar3-self.img_ar2)/(self.img_ar1+self.img_ar3+self.img_ar2))
        return print('Cálculo de índice completo.')

    ###########################################################################
    def plot(self, title = 'Landsat 8 - Index', save = True, 
             output = 'diff_index.TIF', shp = None, cmap = 'gray', cmap_inver = False):
        import matplotlib.pyplot as plt, geopandas as gpd
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1,1, figsize = (10,6))
        
        if cmap_inver == True:
            cmap = plt.cm.get_cmap(cmap).reversed()
        
        img = ax.imshow(self.index, cmap = cmap, vmin = -1, vmax = 1)
        fig.colorbar(img, ax = ax)
        
        show(self.index, transform = self.crs, title= title, ax = ax, cmap = cmap)
        
        if shp != None:
            shape = gpd.read_file(shp)  
            shape.boundary.plot(ax = ax, color = cmap, markersize=4.5)
        if save == True:
            plt.savefig(output, dpi = 300)

###############################################################################
class Stack_L8:
    def __init__(self, img, output = 'stack.tif'):
        import rasterio as rio
        with rio.open(img[0]) as src0:
            meta = src0.meta

        meta.update(count = len(img))

        with rio.open(output, 'w', **meta) as dst:
            for id, layer in enumerate(img, start=1):
                with rio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))
        
        return print('Empaquetamiento finalizado.')

###############################################################################
class US_Class:
    def __init__(self, img):
        self.img = img
        import rasterio as rio, numpy as np
        
        self.img_ar = rio.open(self.img).read()
        self.crs = rio.open(self.img).transform
        self.profile = rio.open(self.img).profile
        self.shape = rio.open(self.img).shape
        
        dstack = np.dstack((self.img_ar[0], self.img_ar[1],
                            self.img_ar[2], self.img_ar[3],
                            self.img_ar[4], self.img_ar[5]))
        nrows, ncols, nbands = dstack.shape
        self.img_ar = dstack.reshape((nrows*ncols, nbands))
            
        return print('La imágenes cargadas correctamente.')
    
    def Elbow(self, n_k = 10, max_i = 150, output = 'elbow_kmean.jpg'):
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        wcss = []
        for i in range(1,n_k):
            km = KMeans(n_clusters = i,
                        init = 'k-means++')
            km.fit(self.img_ar)
            wcss.append(km.inertia_)
        
        ax.plot(range(1, n_k), wcss)
        ax.set_ylabel('Distancia media\nobservación-centroide')
        ax.set_xlabel('Valor de K')
        
        plt.savefig(output, dpi = 300)

    def Kmeans(self, k, max_i = 150, output = 'cluster_kmean.jpg', cmap = 'hsv'):
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        km = KMeans(n_clusters = k,
                    init = 'k-means++',
                    max_iter = max_i)
        
        km.fit(self.img_ar)
        print(f'Centroides: {km.cluster_centers_}')
        
        class_ = km.labels_
        class_ = class_.reshape(self.shape)
        
        fig, ax = plt.subplots(1, 1, figsize = (6,6))
        ax.imshow(class_, cmap = cmap)
        ax.set_title('Clasificación no supervisada - Tamshiyacu')
        plt.savefig(output, dpi = 300)

###########
def T_Surface(NDVI, BT):
    import numpy as np
    Pv = (NDVI - np.amin(NDVI))/(np.amax(NDVI) - np.amin(NDVI))
    e = 0.004*Pv + 0.986
    LST = BT/(1 + (0.00115*BT/1.4388)*np.log(e))
    return LST

def correction(array, metadata, band, plot = 'ps'):
    file = open(metadata, 'r', encoding='utf-8')
    
    # Extracción de datos de la metadata
    ML_index = range(165,176); ML = []; AL_index = range(176,187); AL = []
    MaxL_index = range(97, 119, 2); MaxL = []; MinL_index = range(98, 119, 2); MinL = []
    Mp_index = range(186,196); Mp = []; Ap_index = range(196,205); Ap = []
    Maxp_index = range(121, 139, 2); Maxp = []; Minp_index = range(122, 139, 2); Minp = []
    K1_index = range(207, 210, 2); K1 = []; K2_index = range(208, 211, 2); K2 = []
    Thse_index = 76; Date_index = 4
    
    for position, line in enumerate(file):
        if position in ML_index:
            ML.append(float(str(line[-12:-1].replace('=', ''))))   # Multiplicativo radiancia
        elif position in AL_index:
            AL.append(float(str(line[-10:-1].replace('=', ''))))   # Aditivo radiancia
        elif position in MaxL_index:
            MaxL.append(float(str(line[-10:-1].replace('=', '')))) # Radiancia máxima
        elif position in MinL_index:
            MinL.append(float(str(line[-10:-1].replace('=', '')))) # Radiancia mínima
        elif position in Mp_index:
            Mp.append(float(str(line[-12:-1].replace('=', ''))))   # Multiplicativo reflectancia
        elif position in Ap_index:
            Ap.append(float(str(line[-10:-1].replace('=', ''))))   # Aditivo reflectancia
        elif position in Maxp_index:
            Maxp.append(float(str(line[-10:-1].replace('=', '')))) # Reflectancia máxima
        elif position in Minp_index:
            Minp.append(float(str(line[-10:-1].replace('=', '')))) # Reflectancia 
        elif position in K1_index:
            K1.append(float(str(line[-12:-1].replace('=', ''))))   # Constante K1
        elif position in K2_index:
            K2.append(float(str(line[-12:-1].replace('=', ''))))   # Constante K2
        elif position == Thse_index:
            Thse = float(str(line[-12:-1]))                        # Elevación solar
        elif position == Date_index:
            Date = int(str(line[-10:-7]))                          # Fecha juliana
    file.close()
    
    ########## Corrección atmosférica
    ### 1. Calibración radiométrica: ND -> radiancia
    ### 2. Cálculo de reflectancia: radiancia -> reflectancia aparente
    
    from numpy import amin, pi, cos, sin
    LA = array*ML[band-1] + AL[band-1]      # (1) Radiancia espectral del sensor
    
    #pA = (array*Mp[band-3] + Ap[band-3])/sin(Thse*pi/180) # (2.1) Reflectancia TOA
    
    d = 1 - 0.0167*cos((Date-3)*2*pi/365)            # Distancia Tierra-Sol
    Lmin = amin(array)*ML[band-1] + AL[band-1] # Radiancia del menor número digital
    ESUNA = pi*d*MaxL[band-1]/Maxp[band-1]           # Irradiancia Media Solar exo-atmosférica
    LDOS1 = 0.01*ESUNA*sin(Thse*pi/180)/(pi*d**2)    # Radiancia del objeto oscuro
    Lp = Lmin - LDOS1                                # Path radiance. Efecto bruma
    
    ps = pi*(LA - Lp)*d**2/(ESUNA*sin(Thse*pi/180)) # (2.2) Reflectancia en superficie
    
    return ps

"""
# índices espectrales
https://pro.arcgis.com/es/pro-app/latest/help/data/imagery/indices-gallery.htm
http://www.gisandbeers.com/listado-indices-espectrales-sentinel-landsat/
https://eos.com/es/blog/indices-de-vegetacion/
https://medium.com/aerial-acuity/identifying-crop-variability-whats-the-difference-between-ndvi-false-ndvi-and-vari-plant-health-98c380381a33
https://help.dronedeploy.com/hc/en-us/articles/1500004860841-Understanding-NDVI
https://www.usna.edu/Users/oceano/pguth/md_help/html/norm_sat.htm
https://www.researchgate.net/publication/315669759_Forest_canopy_density_assessment_using_different_approaches_-_Review
https://acolita.com/lista-de-indices-espectrales-en-sentinel-2-y-landsat/
"""  