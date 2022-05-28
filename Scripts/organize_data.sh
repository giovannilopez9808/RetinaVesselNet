cd ../Data/
echo "Organización train"
cd train/image
mkdir data
mv *.jpg data
cd ..
mkdir high_contrast/data -p
cd label 
mkdir data
mv *.png data
cd ../..
echo "Organización validate"
cd validate/image
mkdir data
mv *.jpg data
cd ..
mkdir high_contrast/data -p
cd label 
mkdir data
mv *.png data
cd ../..
echo "Organización test"
cd test/image
mkdir data
mv *.jpg data
cd ..
mkdir high_contrast/data -p
cd label 
mkdir data
mv *.jpg data
cd ../..

