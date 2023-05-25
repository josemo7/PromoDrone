import json
import boto3
import os
import cv2
import numpy as np

import csv

#from flatten_json import flatten 
s3=boto3.resource('s3')
target_bucket='outputpromodrone' # please update with the S3 bucket you will be sending files to 

def lambda_handler(event, context):
    print(event)
    
    print('Loading function')
    source_image=event["Records"][0]["s3"]["object"]["key"]
    source_bucket= event["Records"][0]["s3"]["bucket"]["name"]
    print(source_image)


    s3_object = s3.Object(source_bucket,source_image)
    s3_response = s3_object.get()
    file_content = s3_response['Body'].read()
    
    np_array = np.frombuffer(file_content, np.uint8)
    image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
    client=boto3.client('rekognition')
    
    response = client.detect_faces(Image={'S3Object': {'Bucket': source_bucket, 'Name': source_image}},
        Attributes=['ALL'])

    imgHeight,imgWidth =  image_np.shape[:2]
    
    counter = 1

    for faceDetail in response['FaceDetails']:
        box = faceDetail['BoundingBox']
        left = imgWidth * box['Left']
        top = imgHeight * box['Top']
        width = imgWidth * box['Width']
        height = imgHeight * box['Height']       
    
        points = (
            (left,top),
            (left + width, top),
            (left + width, top + height),
            (left , top + height),
            (left, top)
        )
        top_line= (int(left), int(top))
        bot_line= (int(left + width),int(top + height))
        cv2.rectangle(image_np,top_line,bot_line , (0, 255, 0), 5)

        position = (int(left),int(top+height))
        image = cv2.putText(image_np,str(counter), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3) #font stroke)
        counter = counter+1

    cv2.imwrite("/tmp/image.jpg",image)


    s3_client = boto3.client("s3")
    put = s3_client.put_object(Bucket=target_bucket, Key='files/'+source_image, Body=open("/tmp/image.jpg","rb").read())
    put = s3_client.put_object(Bucket=target_bucket, Key='InputTestImage.jpeg', Body=open("/tmp/image.jpg","rb").read())
    print(put)


    #CREATION OF CSV
    lstpnumber = []
    lstAgeMin = []
    lstAgeMax = []
    lstSmile = []
    lstEyeGlasses = []
    lstSunGlasses = []
    lstGender = []
    lstBeard = []
    lstMustache = []
    lstEyesOpen = []
    lstMouthOpen = []
    lstEmotion= []

    for face in response['FaceDetails']:
        lstAgeMin.append(str(face['AgeRange']['Low']))
        lstAgeMax.append(str(face['AgeRange']['High']))
        lstSmile.append(str(face['Smile']['Value']))
        lstEyeGlasses.append(str(face['Eyeglasses']['Value']))
        lstSunGlasses.append(str(face['Sunglasses']['Value']))
        lstGender.append(str(face['Gender']['Value']))
        lstBeard.append(str(face['Beard']['Value']))
        lstMustache.append(str(face['Mustache']['Value']))
        lstEyesOpen.append(str(face['EyesOpen']['Value']))
        lstMouthOpen.append(str(face['MouthOpen']['Value']))
        lstEmotion.append(str(face['Emotions'][0]['Type']).capitalize())
        
    lstpnumber.extend(range(1,len(lstAgeMin)+1))



    # field names 
    fields = ['Face Number', 'AgeMin','AgeMax','Smile', 'EyeGlasses','SunGlasses','Gender', 'Beard', 'Mustache', 'EyesOpen', 'MouthOpen', 'Emotion'] 
    # data rows of csv file 
    rows_asc = [lstpnumber, lstAgeMin,lstAgeMax,lstSmile, lstEyeGlasses,lstSunGlasses,lstGender,lstBeard,lstMustache,lstEyesOpen,lstMouthOpen,lstEmotion] 
    rows = np.transpose(rows_asc)
    
    with open('/tmp/response.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(rows)

    bucket="outputpromodrone"
    file_name = '/tmp/response.csv'
    object_name = '/Response/response.csv'
    #object_name = file_name
    #'Responses/'+source_image[:-5]+'_response.csv'

    name_file='InputTestImage.jpeg'
    #put = s3_client.put_object(Bucket=bucket, Key=file_name, Body=open(object_name,"rb").read())
    put = s3_client.put_object(Bucket=target_bucket, Key="files/"+source_image+".csv", Body=open(file_name,"rb").read())
    put = s3_client.put_object(Bucket=target_bucket, Key= name_file+".csv", Body=open(file_name,"rb").read())
    print(put)
    
    return {
        
        
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }