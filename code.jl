using Images
using DataFrames
using CSV
using Printf
using Statistics
using DecisionTree

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo[:ID])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = load(nameFile)
  
  temp = Gray.(img)
  x[index, :] = reshape(temp, 1, imageSize)
 end
 return x
end

imageSize  = 400

path = "street-view-getting-started-with-julia"

labelsInfoTrain = CSV.read("$(path)/trainLabels.csv")

xTrain = read_data("train", labelsInfoTrain, imageSize, path)

labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv")

xTest = read_data("test", labelsInfoTest, imageSize, path)

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = Int.(map(x -> x[1], labelsInfoTrain[:Class]))

#Convert to array
yTrain = convert(Array, yTrain)

println("Start training")
model = build_forest(yTrain, xTrain, 20, 100, 1.0)
println("End training")

#Get predictions for test data
predTest = apply_forest(model, xTest)

#Convert integer predictions to character
labelsInfoTest[:Class] = Char.(predTest)

#Save predictions
CSV.write("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',', append=false)

#remove # to get accuracy
#accuracy = nfoldCV_forest(yTrain, xTrain, 20, 100, 10, 1.0);
#println("4 fold accuracy: $(mean(accuracy))")
