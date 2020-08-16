# Instruction to Create a PowerApps connector.

### Go to [this link](https://Make.powerapps.com) and Data -> Customer Connectors.

![powerapps](/Imges/Powerapps1.png)

### Click on new connector, import Open APi File and give you connector a name

![powerapps](/Imges/Powerapps2.png)

### Select the Custom Control from the files you downloaded from Github 

![powerapps](/Imges/Powerapps3.png)
![powerapps](/Imges/Powerapps4.png)

Upload some icon and click on create connector. After the connector is created now you can test it with API KEY 
“Bearer 4dSyxGWssgFLdaqc52OTXJGuMlZ9889h” Click on Test , New Connection.

![powerapps](/Imges/Powerapps5.png)

### Pass sample covid xray to body of the request https://qnacovdi19app-bot.azurewebsites.net/image/Covid.png

![powerapps](/Imges/Powerapps6.png)

### Click on Test Operation and you should see a result like below

![powerapps](/Imges/Powerapps7.png)

### Now go to flow.microsoft.com

![powerapps](/Imges/Powerapps8.png)

### Click on Import and choose the flow files to import both of them

![powerapps](/Imges/Powerapps9.png)

### Select the connector and connections

![powerapps](/Imges/Powerapps10.png)

### Click on Import
#### On The Second Flow, make sure to create OneDrive and outlook connections, and then import

![powerapps](/Imges/Powerapps11.png)

### After Successful import you will see below screen
![powerapps](/Imges/Powerapps12.png)

### So far we have configured our connector + Microsoft Flows
Now it is time to import our PowerApp for XRAY Test. Go to make.powerapps.com
Click on Import CanvasApp

![powerapps](/Imges/Powerapps13.png)

### Choose the PowerApp downloaded from Github from the files

![powerapps](/Imges/Powerapps14.png)

### Map all the required components to your environment and click import.Once your app is successfully imported, open it up by clicking on Play as below

![powerapps](/Imges/Powerapps15.png)

### If PowerApp Prompts you that this app needs premium license for using these components, just click on next and start trial
Application needs to use custom components, and custom connectors in Power Platform world is something premium, hence the notification.
After that click allow for all the connections, 

![powerapps](/Imges/Powerapps16.png)

### Now you have your Sample App to test Covid scenarios thru Chest XRAY Images

![powerapps](/Imges/Powerapps17.png)

#### That’s it ! Feel free to edit the app and connectors.


