from osgeo import gdal
import numpy as np
from osgeo import osr
import operator
from multiprocessing.dummy import Pool as ThreadPool
import time


def arr2raster(arr, raster_file, prj=None, trans=None):
    if 'int8' == arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' == arr.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'uint16' == arr.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'int32' == arr.dtype.name:
        datatype = gdal.GDT_Int32
    else:
        datatype = gdal.GDT_Float64

    if len(arr.shape) == 3:
        im_bands, im_height, im_width = arr.shape
    else:
        im_bands, (im_height, im_width) = 1, arr.shape

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, im_width, im_height, im_bands, datatype, options=["TILED=YES", "COMPRESS=LZW"])

    if prj:
        dst_ds.SetProjection(prj)  
    if trans:
        dst_ds.SetGeoTransform(trans)  

    if im_bands == 1:
        dst_ds.GetRasterBand(1).WriteArray(arr)
    else:
        for ii in range(im_bands):
            dst_ds.GetRasterBand(ii + 1).WriteArray(arr[ii])

    del dst_ds
    # dst_ds.FlushCache()
    # dst_ds = None
    print("successfully convert array to raster")


def ku7(y, cNew, LFA, LFB):
    L, M0, P1, M1, N1, P2, M2, N2, P3, M3, N3, P4, M4, N4, P5, M5, N5 = y
    V0 = cNew[0]
    K0 = cNew[1]
    kl = cNew[2]
    cue0 = cNew[3]
    kb0 = cNew[4]
    V1 = cNew[5]
    K1 = cNew[6]
    cuep1 = cNew[7]
    cuen1 = cNew[8]
    kb1 = cNew[9]
    kd1 = cNew[10]
    V2 = cNew[11]
    K2 = cNew[12]
    cuep2 = cNew[13]
    cuen2 = cNew[14]
    kb2 = cNew[15]
    kd2 = cNew[16]
    V3 = cNew[17]
    K3 = cNew[18]
    cuep3 = cNew[19]
    cuen3 = cNew[20]
    kb3 = cNew[21]
    kd3 = cNew[22]
    V4 = cNew[23]
    K4 = cNew[24]
    cuep4 = cNew[25]
    cuen4 = cNew[26]
    kb4 = cNew[27]
    kd4 = cNew[28]
    V5 = cNew[29]
    K5 = cNew[30]
    cuep5 = cNew[31]
    cuen5 = cNew[32]
    kb5 = cNew[33]

    r1 = 0.62
    r2 = 0.239
    r3 = 0.092
    r4 = 0.0354
    r5 = 0.0136
    out = [LFA - V0 * M0 * L / (K0 + L) - kl * L,
           cue0 * (V0 * M0 * L / (K0 + L)) - kb0 * M0,

           r1 * LFB + kl * L + kb0 * M0 - V1 * M1 * P1 / (K1 + P1) - kd1 * P1,
           cuep1 * (V1 * M1 * P1 / (K1 + P1)) + cuen1 * (V1 * M1 * N1 / (K1 + N1)) - kb1 * M1 - kd1 * M1,
           kb1 * M1 - V1 * M1 * N1 / (K1 + N1) - kd1 * N1,

           r2 * LFB + kd1 * P1 - V2 * M2 * P2 / (K2 + P2) - kd2 * P2,
           kd1 * M1 + cuep2 * (V2 * M2 * P2 / (K2 + P2)) + cuen2 * (V2 * M2 * N2 / (K2 + N2)) - kb2 * M2 - kd2 * M2,
           kd1 * N1 + kb2 * M2 - V2 * M2 * N2 / (K2 + N2) - kd2 * N2,

           r3 * LFB + kd2 * P2 - V3 * M3 * P3 / (K3 + P3) - kd3 * P3,
           kd2 * M2 + cuep3 * (V3 * M3 * P3 / (K3 + P3)) + cuen3 * (V3 * M3 * N3 / (K3 + N3)) - kb3 * M3 - kd3 * M3,
           kd2 * N2 + kb3 * M3 - V3 * M3 * N3 / (K3 + N3) - kd3 * N3,

           r4 * LFB + kd3 * P3 - V4 * M4 * P4 / (K4 + P4) - kd4 * P4,
           kd3 * M3 + cuep4 * (V4 * M4 * P4 / (K4 + P4)) + cuen4 * (V4 * M4 * N4 / (K4 + N4)) - kb4 * M4 - kd4 * M4,
           kd3 * N3 + kb4 * M4 - V4 * M4 * N4 / (K4 + N4) - kd4 * N4,

           r5 * LFB + kd4 * P4 - V5 * M5 * P5 / (K5 + P5),
           kd4 * M4 + cuep5 * (V5 * M5 * P5 / (K5 + P5)) + cuen5 * (V5 * M5 * N5 / (K5 + N5)) - kb5 * M5,
           kd4 * N4 + kb5 * M5 - V5 * M5 * N5 / (K5 + N5)]
    return out


def run_model(cNew, Litterfall, LFB_LF, x0):
    x = np.zeros([17, Nt + 1], dtype=float) 

    x[:, 0] = x0  
    for i2 in range(1, Nt + 1):
        outku7 = ku7(x[:, i2-1], cNew, Litterfall[i2-1] * (1-LFB_LF),  Litterfall[i2-1] * LFB_LF)
        x[:, i2] = outku7 + x[:, i2-1]

    littersimu = x[0, :] + x[1, :]
    socsimu_20 = x[2, :] + x[3, :] + x[4, :]
    socsimu_40 = x[5, :] + x[6, :] + x[7, :]
    socsimu_60 = x[8, :] + x[9, :] + x[10, :]
    socsimu_80 = x[11, :] + x[12, :] + x[13, :]
    socsimu_100 = x[14, :] + x[15, :] + x[16, :]

    return littersimu, socsimu_20, socsimu_40, socsimu_60, socsimu_80, socsimu_100


def de_func2(X, a, b, c):
    x1, x2 = X
    return a*(x2-x1)+((b-a)/(-c)) * (np.exp(-c * x2) - np.exp(-c * x1))


def depth_soc_20(d_socQA, a, b, c, soc05, soc515, soc1530, soc3060, soc60100):
    if d_socQA == 1:
        xx1 = np.array([0, 20, 40, 60, 80])
        xx2 = np.array([20, 40, 60, 80, 100])
        soc_out = de_func2((xx1, xx2), a, b, c)
    else:
        soc_out = np.array([soc05 + soc515 + soc1530 / 3,
                            soc1530 / 3 * 2 + soc3060 / 3,
                            soc3060 - soc3060 / 3,
                            soc60100 / 2,
                            soc60100 / 2])
    return soc_out / 10


def rowcol_to_lonlat(extend, row, col):
    lon = extend[0] + col * extend[1] + row * extend[2]
    lat = extend[3] + col * extend[4] + row * extend[5]
    return lon, lat


def rowcol_to_xy(extend, row, col):
    #
    x = extend[0] + col * extend[1] + row * extend[2]
    y = extend[3] + col * extend[4] + row * extend[5]
    return x, y


def xy_to_lonlat(gcs, pcs, x, y):
    #
    ct = osr.CoordinateTransformation(pcs, gcs)
    lat, lon, _ = ct.TransformPoint(x, y)
    #
    return lon, lat


def lonlat_to_rowcol(extend, lon, lat):
    #
    col = int((lon - extend[0] + extend[1] / 2) / extend[1])  
    row = int((lat - extend[3] + extend[5] / 2) / extend[5]) 
    return row, col


def forjj(ii):
    # print(ii)
    XT_point = [0] * 48  
    for jj in range(lQS_ID, lQS_ID + lzk_range2):  # range(land_width)
        # print('================')
        land_cover = land_dataset.GetRasterBand(1).ReadAsArray(jj, ii, 1, 1)

        if land_cover[0, 0] == 20:
            veg_type = 'Forest'
            param_0_20 = param_QM_0_20
            param_20_40 = param_QM_20_40
            param_40_60 = param_QM_40_60
            param_60_80 = param_QM_60_80
            param_80_100 = param_QM_80_100
        elif land_cover[0, 0] == 30:
            veg_type = 'Grass'
            param_0_20 = param_CD_0_20
            param_20_40 = param_CD_20_40
            param_40_60 = param_CD_40_60
            param_60_80 = param_CD_60_80
            param_80_100 = param_CD_80_100
        elif land_cover[0, 0] == 40:
            veg_type = 'Shrub'
            param_0_20 = param_GM_0_20
            param_20_40 = param_GM_20_40
            param_40_60 = param_GM_40_60
            param_60_80 = param_GM_60_80
            param_80_100 = param_GM_80_100
        else:
            continue

        xx, yy = rowcol_to_xy(land_extend, ii, jj)
        lon, lat = xy_to_lonlat(land_gcs, land_pcs, xx, yy)

        depth_mask_row, depth_mask_col = lonlat_to_rowcol(depth_mask_extend, lon, lat)
        depth_xs_row, depth_xs_col = lonlat_to_rowcol(depth_xs_extend, lon, lat)
        SOC0_5_row, SOC0_5_col = lonlat_to_rowcol(depth_SOC0_5_extend, lon, lat)
        SOC5_15_row, SOC5_15_col = lonlat_to_rowcol(depth_SOC5_15_extend, lon, lat)
        SOC15_30_row, SOC15_30_col = lonlat_to_rowcol(depth_SOC15_30_extend, lon, lat)
        SOC30_60_row, SOC30_60_col = lonlat_to_rowcol(depth_SOC30_60_extend, lon, lat)
        SOC60_100_row, SOC60_100_col = lonlat_to_rowcol(depth_SOC60_100_extend, lon, lat)
        LF_row, LF_col = lonlat_to_rowcol(LF_extend, lon, lat)
        TG_row, TG_col = lonlat_to_rowcol(TG_extend, lon, lat)

        LF = np.zeros((lf_bands))
        for i in range(1, lf_bands + 1):
            band = LF_dataset.GetRasterBand(i)
            LF[i - 1] = band.ReadAsArray(LF_col, LF_row, 1, 1)

        depth_aa = xs_dataset.GetRasterBand(1).ReadAsArray(depth_xs_col, depth_xs_row, 1, 1)
        depth_bb = xs_dataset.GetRasterBand(2).ReadAsArray(depth_xs_col, depth_xs_row, 1, 1)
        depth_cc = xs_dataset.GetRasterBand(3).ReadAsArray(depth_xs_col, depth_xs_row, 1, 1)

        depth_mask = mask_dataset.GetRasterBand(1).ReadAsArray(depth_mask_col, depth_mask_row, 1, 1)
        depth_SOC0_5 = SOC0_5_dataset.GetRasterBand(1).ReadAsArray(SOC0_5_col, SOC0_5_row, 1, 1)
        depth_SOC5_15 = SOC5_15_dataset.GetRasterBand(1).ReadAsArray(SOC5_15_col, SOC5_15_row, 1, 1)
        depth_SOC15_30 = SOC15_30_dataset.GetRasterBand(1).ReadAsArray(SOC15_30_col, SOC15_30_row, 1, 1)
        depth_SOC30_60 = SOC30_60_dataset.GetRasterBand(1).ReadAsArray(SOC30_60_col, SOC30_60_row, 1, 1)
        depth_SOC60_100 = SOC60_100_dataset.GetRasterBand(1).ReadAsArray(SOC60_100_col, SOC60_100_row, 1, 1)

        TG_yr = TG_dataset.GetRasterBand(1).ReadAsArray(TG_col, TG_row, 1, 1)
        # TG_yr_sd = TG_dataset.GetRasterBand(2).ReadAsArray(TG_col, TG_row, 1, 1)

        temp = [land_cover[0, 0], depth_mask[0, 0], depth_aa[0, 0], depth_bb[0, 0], depth_cc[0, 0], depth_SOC0_5[0, 0],
                depth_SOC5_15[0, 0], depth_SOC15_30[0, 0], depth_SOC30_60[0, 0], depth_SOC60_100[0, 0], TG_yr[0, 0]]
        temp.extend(LF.tolist())
        temp2 = np.array(temp)
        if (len(temp2[np.isnan(temp2)]) > 0) or (len(temp2[(temp2 == 32767)]) > 0) or (len(temp2[(temp2 == -99)]) > 0):
            continue

        if operator.eq(XT_point, temp):

            outdata1[:, ii - hQS_ID, jj - lQS_ID] = outdata1[:, ii - hQS_ID, jj - lQS_ID - 1]
            outdata2[:, ii - hQS_ID, jj - lQS_ID] = outdata2[:, ii - hQS_ID, jj - lQS_ID - 1]
            outdata3[:, ii - hQS_ID, jj - lQS_ID] = outdata3[:, ii - hQS_ID, jj - lQS_ID - 1]
            outdata4[:, ii - hQS_ID, jj - lQS_ID] = outdata4[:, ii - hQS_ID, jj - lQS_ID - 1]
            outdata5[:, ii - hQS_ID, jj - lQS_ID] = outdata5[:, ii - hQS_ID, jj - lQS_ID - 1]

            continue
        else:
            XT_point = temp

        ini_SOC = depth_soc_20(depth_mask[0, 0], depth_aa[0, 0], depth_bb[0, 0], depth_cc[0, 0], depth_SOC0_5[0, 0],
                               depth_SOC5_15[0, 0], depth_SOC15_30[0, 0], depth_SOC30_60[0, 0], depth_SOC60_100[0, 0])

        ini_SOC = ini_SOC / 4

        if veg_type == 'Forest':
            litter0 = ini_SOC[0] * 0.95
        elif veg_type == 'Grass':
            litter0 = ini_SOC[0] * 0.14
        else:
            litter0 = ini_SOC[0] * 0.66
        # print(ini_SOC[0])

        LF_point = LF
        LF_point = np.divide(LF_point, 1000)  

        TG_yr_point = round(TG_yr[0, 0])

        LF2 = LF_point[(TG_yr_point - 1982):]
        LF_list_all = np.append(LF2, np.array([LF2[-1]] * (Nt - (2018 - TG_yr_point))))

        x0 = [litter0, 0.0023 * litter0,  # L, M0
              ini_SOC[0] - ini_SOC[0] * 0.215, ini_SOC[0] * 0.015, ini_SOC[0] * 0.2,  # P1, M1, N1
              ini_SOC[1] - ini_SOC[1] * 0.215, ini_SOC[1] * 0.015, ini_SOC[1] * 0.2,  # P2, M2, N2
              ini_SOC[2] - ini_SOC[2] * 0.215, ini_SOC[2] * 0.015, ini_SOC[2] * 0.2,  # P3, M3, N3
              ini_SOC[3] - ini_SOC[3] * 0.215, ini_SOC[3] * 0.015, ini_SOC[3] * 0.2,  # P4, M4, N4
              ini_SOC[4] - ini_SOC[4] * 0.215, ini_SOC[4] * 0.015, ini_SOC[4] * 0.2]  # P5, M5, N5

        littersimu1, socsimu_20, _, _, _, _ = run_model(param_0_20, LF_list_all, LFB_LF, x0)
        littersimu2, _, socsimu_40, _, _, _ = run_model(param_20_40, LF_list_all, LFB_LF, x0)
        littersimu3, _, _, socsimu_60, _, _ = run_model(param_40_60, LF_list_all, LFB_LF, x0)
        littersimu4, _, _, _, socsimu_80, _ = run_model(param_60_80, LF_list_all, LFB_LF, x0)
        littersimu5, _, _, _, _, socsimu_100 = run_model(param_80_100, LF_list_all, LFB_LF, x0)

        socsimu_20 = np.round(socsimu_20 * 1000).astype(np.uint16)
        socsimu_40 = np.round(socsimu_40 * 1000).astype(np.uint16)
        socsimu_60 = np.round(socsimu_60 * 1000).astype(np.uint16)
        socsimu_80 = np.round(socsimu_80 * 1000).astype(np.uint16)
        socsimu_100 = np.round(socsimu_100 * 1000).astype(np.uint16)

        outdata1[0:(TG_yr_point - 1982), ii - hQS_ID, jj - lQS_ID] = [socsimu_20[0]] * (TG_yr_point - 1982)
        outdata1[(TG_yr_point - 1982): 37, ii - hQS_ID, jj - lQS_ID] = socsimu_20[(TG_yr_point - 1982): 37]
        outdata1[37, ii - hQS_ID, jj - lQS_ID] = socsimu_20[-1]

        outdata2[0:(TG_yr_point - 1982), ii - hQS_ID, jj - lQS_ID] = [socsimu_40[0]] * (TG_yr_point - 1982)
        outdata2[(TG_yr_point - 1982): 37, ii - hQS_ID, jj - lQS_ID] = socsimu_40[(TG_yr_point - 1982): 37]
        outdata2[37, ii - hQS_ID, jj - lQS_ID] = socsimu_40[-1]

        outdata3[0:(TG_yr_point - 1982), ii - hQS_ID, jj - lQS_ID] = [socsimu_60[0]] * (TG_yr_point - 1982)
        outdata3[(TG_yr_point - 1982): 37, ii - hQS_ID, jj - lQS_ID] = socsimu_60[(TG_yr_point - 1982): 37]
        outdata3[37, ii - hQS_ID, jj - lQS_ID] = socsimu_60[-1]

        outdata4[0:(TG_yr_point - 1982), ii - hQS_ID, jj - lQS_ID] = [socsimu_80[0]] * (TG_yr_point - 1982)
        outdata4[(TG_yr_point - 1982): 37, ii - hQS_ID, jj - lQS_ID] = socsimu_80[(TG_yr_point - 1982): 37]
        outdata4[37, ii - hQS_ID, jj - lQS_ID] = socsimu_80[-1]

        outdata5[0:(TG_yr_point - 1982), ii - hQS_ID, jj - lQS_ID] = [socsimu_100[0]] * (TG_yr_point - 1982)
        outdata5[(TG_yr_point - 1982): 37, ii - hQS_ID, jj - lQS_ID] = socsimu_100[(TG_yr_point - 1982): 37]
        outdata5[37, ii - hQS_ID, jj - lQS_ID] = socsimu_100[-1]


if __name__ == "__main__":
    indir = '../data/'
    outDir1 = 'F:/output_SOC_SimuRaster/depth_0_20cm/'
    outDir2 = 'F:/output_SOC_SimuRaster/depth_20_40cm/'
    outDir3 = 'F:/output_SOC_SimuRaster/depth_40_60cm/'
    outDir4 = 'F:/output_SOC_SimuRaster/depth_60_80cm/'
    outDir5 = 'F:/output_SOC_SimuRaster/depth_80_100cm/'

    # read data============================================================================
    outDir = indir + 'MCMC'
    ecotype = 'shrub'
    param_GM_0_20 = np.loadtxt(outDir + '/' + ecotype + '0_20.txt')
    param_GM_20_40 = np.loadtxt(outDir + '/' + ecotype + '20_40.txt')
    param_GM_40_60 = np.loadtxt(outDir + '/' + ecotype + '40_60.txt')
    param_GM_60_80 = np.loadtxt(outDir + '/' + ecotype + '60_80.txt')
    param_GM_80_100 = np.loadtxt(outDir + '/' + ecotype + '80_100.txt')
    ecotype = 'forest'
    param_QM_0_20 = np.loadtxt(outDir + '/' + ecotype + '0_20.txt')
    param_QM_20_40 = np.loadtxt(outDir + '/' + ecotype + '20_40.txt')
    param_QM_40_60 = np.loadtxt(outDir + '/' + ecotype + '40_60.txt')
    param_QM_60_80 = np.loadtxt(outDir + '/' + ecotype + '60_80.txt')
    param_QM_80_100 = np.loadtxt(outDir + '/' + ecotype + '80_100.txt')
    ecotype = 'grass'
    param_CD_0_20 = np.loadtxt(outDir + '/' + ecotype + '0_20.txt')
    param_CD_20_40 = np.loadtxt(outDir + '/' + ecotype + '20_40.txt')
    param_CD_40_60 = np.loadtxt(outDir + '/' + ecotype + '40_60.txt')
    param_CD_60_80 = np.loadtxt(outDir + '/' + ecotype + '60_80.txt')
    param_CD_80_100 = np.loadtxt(outDir + '/' + ecotype + '80_100.txt')

    LF_dataset = gdal.Open(indir + 'litterfall/GLASS_1982_2018_LF_YEAR_500M_HT_Clip.dat')
    lf_width = LF_dataset.RasterXSize
    lf_height = LF_dataset.RasterYSize
    lf_bands = LF_dataset.RasterCount
    LF_extend = LF_dataset.GetGeoTransform()

    xs_dataset = gdal.Open(indir + 'SOC0/Depth_coefficient.tif')
    depth_xs_extend = xs_dataset.GetGeoTransform()

    mask_dataset = gdal.Open(indir + 'SOC0/Depth_coefficientQA.tif')
    depth_mask_extend = mask_dataset.GetGeoTransform()

    # units: t / ha
    SOC0_5_dataset = gdal.Open(indir + 'SOC0/OCSTHA_M_sd1_250m_ll_Clip1.tif')
    depth_SOC0_5_extend = SOC0_5_dataset.GetGeoTransform()

    SOC5_15_dataset = gdal.Open(indir + 'SOC0/OCSTHA_M_sd2_250m_ll_Clip1.tif')
    depth_SOC5_15_extend = SOC5_15_dataset.GetGeoTransform()

    SOC15_30_dataset = gdal.Open(indir + 'SOC0/OCSTHA_M_sd3_250m_ll_Clip1.tif')
    depth_SOC15_30_extend = SOC15_30_dataset.GetGeoTransform()

    SOC30_60_dataset = gdal.Open(indir + 'SOC0/OCSTHA_M_sd4_250m_ll_Clip1.tif')
    depth_SOC30_60_extend = SOC30_60_dataset.GetGeoTransform()

    SOC60_100_dataset = gdal.Open(indir + 'SOC0/OCSTHA_M_sd5_250m_ll_Clip1.tif')
    depth_SOC60_100_extend = SOC60_100_dataset.GetGeoTransform()

    land_dataset = gdal.Open(indir + 'Globeland30m_2020/HT_clip3.tif')
    land_width = land_dataset.RasterXSize
    land_height = land_dataset.RasterYSize

    land_extend = land_dataset.GetGeoTransform()
    land_proj = land_dataset.GetProjection()

    land_pcs = osr.SpatialReference()  
    land_pcs.ImportFromWkt(land_dataset.GetProjection())
    land_gcs = land_pcs.CloneGeogCS()

    TG_dataset = gdal.Open(indir + 'ST_VRPs/st.tif')
    TG_extend = TG_dataset.GetGeoTransform()
    # ============================================================================end

    Nt = 150  
    LFB_LF = 0 

    # Circular partitioning of sub blocks to reduce memory usage
    hzk_range = 1000  
    lzk_range = 1000  
    ksl = int(np.ceil(land_width / lzk_range))  
    ksh = int(np.ceil(land_height / hzk_range))  
    for hh in range(7, 8):  # range(ksh)  ========================
        hzk_range2 = hzk_range
        if hh == (ksh - 1):
            hzk_range2 = land_height - (ksh - 1) * hzk_range
        hQS_ID = hh * hzk_range  # The starting column number of the block calculation

        for vv in range(ksl-1, ksl):  # 20, ksl
            lzk_range2 = lzk_range
            if vv == (ksl - 1):
                lzk_range2 = land_width - (ksl - 1) * lzk_range - 50
            lQS_ID = vv * lzk_range  # The starting column number of the block calculation
            print('h:', hQS_ID, hzk_range2)
            print('l:', lQS_ID, lzk_range2)

            outdata1 = np.full((38, hzk_range2, lzk_range2), 65534, dtype=np.uint16)  # Soil Layer 1
            outdata2 = np.full((38, hzk_range2, lzk_range2), 65534, dtype=np.uint16)  # Soil Layer 2
            outdata3 = np.full((38, hzk_range2, lzk_range2), 65534, dtype=np.uint16)  # Soil Layer 3
            outdata4 = np.full((38, hzk_range2, lzk_range2), 65534, dtype=np.uint16)  # Soil Layer 4
            outdata5 = np.full((38, hzk_range2, lzk_range2), 65534, dtype=np.uint16)  # Soil Layer 5

            # A loop is performed for each pixel
            # The resolution of land cover is the highest, and the cycle is based on land cover
            start = time.time()
            for ii in range(hQS_ID, hQS_ID+hzk_range2):  # range(land_height) 
                forjj(ii)
            end = time.time()
            print('Parallel Computing time:\t', (end - start)/3600)

            # Parallel loop
            # start = time.time()
            # pool = ThreadPool()
            # pool.map(forjj, range(hQS_ID, hQS_ID + hzk_range2))
            # pool.close()
            # pool.join()
            # end = time.time()
            # print('Parallel Computing time:\t', end - start)

            # If the data is all null values, no output 
            if (outdata1 == 65534).all():
                print('No data available')
                continue

            # out file
            out_file1 = outDir1 + '0_20_h' + str(hQS_ID) + '_' + str(
                hQS_ID + hzk_range2 - 1) + '_v' + str(lQS_ID) + '_' + str(lQS_ID + lzk_range2 - 1)+'.tif'
            out_file2 = outDir2 + '20_40_h' + str(hQS_ID) + '_' + str(
                hQS_ID + hzk_range2 - 1) + '_v' + str(lQS_ID) + '_' + str(lQS_ID + lzk_range2 - 1)+'.tif'
            out_file3 = outDir3 + '40_60_h' + str(hQS_ID) + '_' + str(
                hQS_ID + hzk_range2 - 1) + '_v' + str(lQS_ID) + '_' + str(lQS_ID + lzk_range2 - 1)+'.tif'
            out_file4 = outDir4 + '60_80_h' + str(hQS_ID) + '_' + str(
                hQS_ID + hzk_range2 - 1) + '_v' + str(lQS_ID) + '_' + str(lQS_ID + lzk_range2 - 1)+'.tif'
            out_file5 = outDir5 + '80_100_h' + str(hQS_ID) + '_' + str(
                hQS_ID + hzk_range2 - 1) + '_v' + str(lQS_ID) + '_' + str(lQS_ID + lzk_range2 - 1)+'.tif'

            # Projection information of sub blocks
            ht, lt = rowcol_to_xy(land_extend, hQS_ID, lQS_ID)
            zland_extend = (ht, land_extend[1], land_extend[2], lt, land_extend[4], land_extend[5])  # The ancestor cannot be changed
            arr2raster(outdata1, out_file1, prj=land_proj, trans=zland_extend)
            arr2raster(outdata2, out_file2, prj=land_proj, trans=zland_extend)
            arr2raster(outdata3, out_file3, prj=land_proj, trans=zland_extend)
            arr2raster(outdata4, out_file4, prj=land_proj, trans=zland_extend)
            arr2raster(outdata5, out_file5, prj=land_proj, trans=zland_extend)

    del land_dataset
    del LF_dataset
    del xs_dataset
    del mask_dataset
    del SOC0_5_dataset
    del SOC5_15_dataset
    del SOC15_30_dataset
    del SOC30_60_dataset
    del SOC60_100_dataset
    del TG_dataset
