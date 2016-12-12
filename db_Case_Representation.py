#--- DATABASE --- #
import csv

class House:
     #------------------------------------------------
    def __init__(self,row):
        #attr_element = row.split(',')
        self.Id=row[0]
        self.MSSubClass=row[1]
        self.MSZoning=row[2]
        self.LotFrontage=row[3]
        self.LotArea=row[4]
        self.Street=row[5]
        self.Alley=row[6]
        self.LotShape=row[7]
        self.LandContour=row[8]
        self.Utilities=row[9]
        self.LotConfig=row[10]
        self.LandSlope=row[11]
        self.Neighborhood=row[12]
        self.Condition1=row[13]
        self.Condition2=row[14]
        self.BldgType=row[15]
        self.HouseStyle=row[16]
        self.OverallQual=row[17]
        self.OverallCond=row[18]
        self.YearBuilt=row[19]
        self.YearRemodAdd=row[20]
        self.RoofStyle=row[21]
        self.RoofMatl=row[22]
        self.Exterior1st=row[23]
        self.Exterior2nd=row[24]
        self.MasVnrType=row[25]
        self.MasVnrArea=row[26]
        self.ExterQual=row[27]
        self.ExterCond=row[28]
        self.Foundation=row[29]
        self.BsmtQual=row[30]
        self.BsmtCond=row[31]
        self.BsmtExposure=row[32]
        self.BsmtFinType1=row[33]
        self.BsmtFinSF1=row[34]
        self.BsmtFinType2=row[35]
        self.BsmtFinSF2=row[36]
        self.BsmtUnfSF=row[37]
        self.TotalBsmtSF=row[38]
        self.Heating=row[39]
        self.HeatingQC=row[40]
        self.CentralAir=row[41]
        self.Electrical=row[42]
        self.fstFlrSF=row[43] #
        self.sndFlrSF=row[44]#
        self.LowQualFinSF=row[45]
        self.GrLivArea=row[46]
        self.BsmtFullBath=row[47]
        self.BsmtHalfBath=row[48]
        self.FullBath=row[49]
        self.HalfBath=row[50]
        self.Bedroom=row[51]
        self.Kitchen=row[52]
        self.KitchenQual=row[53]
        self.TotRmsAbvGrd=row[54]
        self.Functional=row[55]
        self.Fireplaces=row[56]
        self.FireplaceQu=row[57]
        self.GarageType=row[58]
        self.GarageYrBlt=row[59]
        self.GarageFinish=row[60]
        self.GarageCars=row[61]
        self.GarageArea=row[62]
        self.GarageQual=row[63]
        self.GarageCond=row[64]
        self.PavedDrive=row[65]
        self.WoodDeckSF=row[66]
        self.OpenPorchSF=row[67]
        self.EnclosedPorch=row[68]
        self.thSsnPorch=row[69] #
        self.ScreenPorch=row[70]
        self.PoolArea=row[71]
        self.PoolQC=row[72]
        self.Fence=row[73]
        self.MiscFeature=row[74]
        self.MiscVal=row[75]
        self.MoSold=row[76]
        self.YrSold=row[77]
        self.SaleType=row[78]
        self.SaleCondition=row[79]

houses = []
with open ("dataFromKaggle/train.csv") as f_read:
    csv_reader = csv.reader(f_read)
    for row in csv_reader:
        houses.append(House(row))
print(houses[1].Id)
data = type("Data",(),{})
setattr(data, houses[0].Id, houses[4].Id)
l = data()
print l.Id
