import Pkg
using Images
using CSV
using DataFrames

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo["ID"])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = load(nameFile)
  
  temp = Gray.(img)

  #Transform image matrix to a vector and store
  #it in data matrix
  x[index, :] = reshape(temp, 1, imageSize)
 end
 return x
end

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "."

#Read information about training data , IDs.
labelsInfoTrain = CSV.read("$(path)/trainLabels.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

yTrain = map(x -> x[1], labelsInfoTrain["Class"])

#Convert from character to integer
yTrain = int(yTrain)

xTrain = xTrain'
xTest = xTest'




CSV.write("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',',append = false)

println("Submission file saved in $(path)/juliaKNNSubmission.csv")
