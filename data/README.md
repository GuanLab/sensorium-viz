
#### Retrieve and process the Sensorium data

1. Download data from https://gin.g-node.org/cajal/Sensorium2022  
      
    We only need 5 mice: 21067, 22846, 23343, 23656, 23964. The other two are the held-out test for this challenge. The responses of the test images were hidden.  
    Note that the download speed could be unexpected slow.  

2. Unzip and rename the 5 mice datasets    

    ```bash
    # for example
    unzip static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
    mv static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6/ 21067-10-18/
    ```

3. Merge the repetitions in test images

    ```bash
    # for mouse 21067-10-18
    python create_agg_test.py 21067-10-18
    ```