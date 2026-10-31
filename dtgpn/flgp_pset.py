from strongGPDataType import feex_size1, feex_size2, num_bins, hog_size1, hog_size2, region
from strongGPDataType import ndarray, histdata, radius, no_neigh, method, histtData,kernelSize
from strongGPDataType import coordsY2, windowSize2, indexType1, indexType2, windowSize

import feature_function_mt as fe_fs
import gp_tree

def global_set_nopara():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    pset.addPrimitive(fe_fs.all_feex, [region], histdata, name='Global_FeEx')
    pset.addPrimitive(fe_fs.all_histogram, [region], histdata, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [region], histdata, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [region], histdata, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [region], histdata, name='Global_SIFT')
    return pset

def global_set_ap():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    pset.addPrimitive(fe_fs.all_feex, [region], histdata, name='Global_FeEx')
    pset.addPrimitive(fe_fs.all_histogram, [region], histdata, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [region], histdata, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [region], histdata, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [region], histdata, name='Global_SIFT')
    pset.addPrimitive(fe_fs.averaging, [region, kernelSize], region, name='Averaging')
    pset.addPrimitive(fe_fs.maxP, [region, kernelSize], region, name='MaxPooling')
    pset.addPrimitive(fe_fs.averaging, [region, kernelSize], region, name='Averaging2')
    pset.addPrimitive(fe_fs.maxP, [region, kernelSize], region, name='MaxPooling2')
    return pset

def global_pset():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    pset.addPrimitive(fe_fs.all_feex, [region, feex_size1, feex_size2], histdata, name='Global_FeEx')
    pset.addPrimitive(fe_fs.all_histogram, [region, num_bins], histdata, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [region, hog_size1, hog_size2], histdata, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp_ip, [region, radius, method], histdata, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [region], histdata, name='Global_SIFT')
    return pset

def local_set_nopara():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    pset.addPrimitive(fe_fs.all_feex, [ndarray], histdata, name='Local_FeEx')
    pset.addPrimitive(fe_fs.all_histogram, [ndarray], histdata, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [ndarray], histdata, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [ndarray], histdata, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [ndarray], histdata, name='Local_SIFT')
    # Terminals
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize], ndarray, name='Region_S1')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize2], ndarray, name='Region_S2')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize, windowSize], ndarray, name='Region_R1')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize2, windowSize2], ndarray, name='Region_R2')
    return pset

def local_pset():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    pset.addPrimitive(fe_fs.all_feex, [ndarray, feex_size1, feex_size2], histdata, name='Local_FeEx')
    pset.addPrimitive(fe_fs.all_histogram, [ndarray, num_bins], histdata, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [ndarray, hog_size1, hog_size2], histdata, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp_ip, [ndarray, radius, method], histdata, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [ndarray], histdata, name='Local_SIFT')
    # Terminals
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize], ndarray, name='Region_S1')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize2], ndarray, name='Region_S2')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize, windowSize], ndarray, name='Region_R1')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize2, windowSize2], ndarray, name='Region_R2')
    return pset

def all_pset(Is_RGB):
    if Is_RGB:
        pset = gp_tree.PrimitiveSetTyped('MAIN', [ region, region, region], histdata, prefix='Image')
    else:
        pset = gp_tree.PrimitiveSetTyped('MAIN', [region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    ##pset.addPrimitive(fe_fs.global_gabor,[region],poolingType,name='Global_Gabor')
    pset.addPrimitive(fe_fs.all_feex, [region], histdata, name='Global_FeEx')
    ##pset.addPrimitive(fe_fs.all_glcm,[region],poolingType,name='Global_GLCM')
    pset.addPrimitive(fe_fs.all_histogram, [region], histdata, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [region], histdata, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [region], histdata, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [region], histdata, name='Global_SIFT')
    # local feature extractor
    ##pset.addPrimitive(fe_fs.local_gabor,[ndarray],filterData,name='Local_Gabor')
    pset.addPrimitive(fe_fs.all_feex, [ndarray], histdata, name='Local_FeEx')
    ##pset.addPrimitive(fe_fs.all_glcm,[ndarray],filterData,name='Local_GLCM')
    pset.addPrimitive(fe_fs.all_histogram, [ndarray], histdata, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [ndarray], histdata, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [ndarray], histdata, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [ndarray], histdata, name='Local_SIFT')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize], ndarray, name='Region_S1')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize2], ndarray, name='Region_S2')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize, windowSize], ndarray,
                      name='Region_R1')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize2, windowSize2], ndarray,
                      name='Region_R2')
    return pset


def all_pset_color():
    pset = gp_tree.PrimitiveSetTyped('MAIN', [region, region, region, region], histdata, prefix='Image')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata], histdata, name='root2')
    pset.addPrimitive(fe_fs.root_con, [histdata, histdata, histdata], histdata, name='root3')
    # global feature extractor
    ##pset.addPrimitive(fe_fs.global_gabor,[region],poolingType,name='Global_Gabor')
    pset.addPrimitive(fe_fs.all_feex, [region], histdata, name='Global_FeEx')
    ##pset.addPrimitive(fe_fs.all_glcm,[region],poolingType,name='Global_GLCM')
    pset.addPrimitive(fe_fs.all_histogram, [region], histdata, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [region], histdata, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [region], histdata, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [region], histdata, name='Global_SIFT')
    # local feature extractor
    ##pset.addPrimitive(fe_fs.local_gabor,[ndarray],filterData,name='Local_Gabor')
    pset.addPrimitive(fe_fs.all_feex, [ndarray], histdata, name='Local_FeEx')
    ##pset.addPrimitive(fe_fs.all_glcm,[ndarray],filterData,name='Local_GLCM')
    pset.addPrimitive(fe_fs.all_histogram, [ndarray], histdata, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [ndarray], histdata, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [ndarray], histdata, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [ndarray], histdata, name='Local_SIFT')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize], ndarray, name='Region_S1')
    pset.addPrimitive(fe_fs.regionS, [region, indexType1, indexType2, windowSize2], ndarray, name='Region_S2')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize, windowSize], ndarray,
                      name='Region_R1')
    pset.addPrimitive(fe_fs.regionR, [region, indexType1, indexType2, windowSize2, windowSize2], ndarray,
                      name='Region_R2')
    return pset

