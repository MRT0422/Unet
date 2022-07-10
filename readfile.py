from osgeo import gdal
#  读取图像像素矩阵
#  fileName 图像文件名
def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data