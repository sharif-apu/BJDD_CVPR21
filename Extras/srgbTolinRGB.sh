mkdir -p linRGB
for img in `ls DIV2K_train_HR/*.png`; 
do
echo $img
convert $img  -colorspace RGB -set filename:base "%[basename]" "linRGB/%[filename:base].png"; 
done
