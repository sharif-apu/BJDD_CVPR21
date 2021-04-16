import cv2


def bayerSampler(image):
    img = image.copy()

    # R samler
    img[1::2,1::2, 1:3]=0

    # B sampler
    img[::2,::2, 0:2]=0

    # G12 sampler
    img[::2,1::2, ::2]=0

    # G21 sampler
    img[1::2,::2, ::2]=0

    return img


def quadBayerSampler(image):
    img  = image.copy()
    
    # Quad R 
    img[::4,::4, 1:3]=0
    img[1::4,1::4, 1:3]=0
    img[::4,1::4, 1:3]=0
    img[1::4,::4, 1:3]=0

    # Quad B 
    img[3::4,2::4, 0:2]=0
    img[3::4,3::4, 0:2]=0
    img[2::4,3::4, 0:2]=0
    img[2::4,2::4, 0:2]=0

    #Quad G12
    img[1::4,2::4, ::2]=0
    img[1::4,3::4, ::2]=0
    img[::4,2::4, ::2]=0
    img[::4,3::4, ::2]=0
    
    #Quad G21
    img[2::4,1::4, ::2]=0
    img[3::4,1::4, ::2]=0
    img[2::4,::4, ::2]=0
    img[3::4,::4, ::2]=0


    return img


def dynamicBayerSamplerOpenCV(image, gridSize = 4):

    # Initiating parameters
    img = image.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    row= 0
    col = 0

    while row < rows:
        while col < cols:
            # R sampler (opencv)
            for i in range (row, row + gridSize):
                for j in range (col, col + gridSize):
                    if (i >= rows or j >= cols): 
                        break 
                    img[i,j,1:3] = 0

            # B sampler (opencv)
            for i in range (row + gridSize, row + gridSize + gridSize):
                for j in range (col + gridSize, col + gridSize + gridSize):
                    if (i >= rows or j >= cols):  
                        break 
                    img[i,j,:2] = 0

            # G12 sampler
            for i in range (row, row + gridSize):
                for j in range (col + gridSize, col + gridSize + gridSize):
                    if (i >= rows or j >= cols):  
                        break 
                    img[i,j,::2] = 0

            # G21 sampler
            for i in range (row + gridSize, row + gridSize + gridSize):
                for j in range (col, col + gridSize):
                    if (i >= rows or j >= cols):   
                        break 
                    img[i,j,::2] = 0

            # Updading column index
            col += (gridSize * 2)

        # Initiating column index to iterate over a new row
        col = 0

        # Updading row index
        row += (gridSize * 2)

    return img


def dynamicBayerSampler(image, gridSize = 4):

    # Initiating parameters
    img = image.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    row= 0
    col = 0

    while row < rows:
        while col < cols:
            # R sampler 
            for i in range (row, row + gridSize):
                for j in range (col, col + gridSize):
                    if (i >= rows or j >= cols): 
                        break 
                    img[i,j,:2] = 0

            # B sampler
            for i in range (row + gridSize, row + gridSize + gridSize):
                for j in range (col + gridSize, col + gridSize + gridSize):
                    if (i >= rows or j >= cols):  
                        break 
                    img[i,j,1:3] = 0

            # G12 sampler
            for i in range (row, row + gridSize):
                for j in range (col + gridSize, col + gridSize + gridSize):
                    if (i >= rows or j >= cols):  
                        break 
                    img[i,j,::2] = 0

            # G21 sampler 
            for i in range (row + gridSize, row + gridSize + gridSize):
                for j in range (col, col + gridSize):
                    if (i >= rows or j >= cols):   
                        break 
                    img[i,j,::2] = 0

            # Updading column index
            col += (gridSize * 2)

        # Initiating column index to iterate over a new row
        col = 0

        # Updading row index
        row += (gridSize * 2)

    return img

   

if __name__ == "__main__":
    
    #cv2.waitKey(1000)
    imName = "./3.png"
    image = cv2.imread(imName)
    print (image.shape)
    #img = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    #imgr=i
    imgSampled = quadBayerSampler(image.copy())
    cv2.imwrite("./outputorg.png", image)
    cv2.imwrite("./output.png", imgSampled)
