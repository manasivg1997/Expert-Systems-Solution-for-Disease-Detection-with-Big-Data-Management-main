# Expert-Systems-Solution-for-Disease-Detection-with-Big-Data-Management (Feb 2022)

## About the Project
The key idea of this project is to fully utilize the potential of Big Data Management & Analytics in combination with Machine Learning and Natural Language Processing that belong to the umbrella of Expert Systems. This project is used to suggest the likeliness of having top 10 diseases along with their probabilities based on the extent to which there is a match with the symptoms a person is experiencing. An interactive system using a GUI is built which acts as a platform for the user to find which diseases he/she may be exposed to by providing the symptoms to the System.

1. Natural Language Processing : Handles knowledge extraction & feature generation.
2. Machine Learning : Building models based on the data.
3. Big Data Management & Data Analytics : Processing huge data & Visualizing the inferences.

## Execution steps
Note: Navigate into respective folders
1. Run DataCollection.ipynb to get all diseases & symptoms associated with each disease
2. Run DataPreprocessing.ipynb to extract CSV files from the data
3. Run ModelsAndPlots.ipynb for Model analysis
4. Run GetModelWeights.ipynb to generate weights assoicated with each model that is trained on the dataset
6. Run Top10DiseasesForSymptoms.ipynb to check with sample input
7. Run ProjectEvaluation.ipynb for entire System evaluation
8. To integrate with UI, first setup Database & then the DjangoApp
9. Run DB-Setup.sql and then DiseaseSymptomsData.sql
10. Follow instructions to setup Django
11. On localhost:8000, the project starts execution

## Instructions to set up Django

#### For Mac:
1. Navigate to env —> cd env
2. Activate the virtual environment —> source env/bin/activate —> Before the username, you should see (env)
3. Now back to project folder —> cd ..

#### For Windows:
1. Navigate to env/Scripts —> cd env/Scripts
2. Activate the virtual environment —> activate —> Before the username, you should see (env)
3. Now back to project folder —> cd ..    cd ..

#### Common Installations:
Do the following installations only after the virtual environment is activated
1. pip install pymysql
2. pip install Django
3. Download & install xampp

#### To run the project:
1. Start xampp server
2. Go to http://localhost/phpmyadmin/
3. Import —> Run the swiftdiagnosis.sql file (Do this only once)
4. Navigate to project folder & then --> python -m venv env (For Mac users)
5. python manage.py runserver
6. To stop, use Ctrl+C
7. http://localhost:8000 to view the UI up & running

## Architecture
<p>
<img width="550" alt="Screen Shot 2022-05-17 at 4 13 03 PM" src="https://user-images.githubusercontent.com/28973352/168911048-77854141-2097-4d2b-bf22-a814aa22b68b.png"> 
<img width="550" alt="Screen Shot 2022-05-17 at 4 09 28 PM" src="https://user-images.githubusercontent.com/28973352/168910669-e07657c6-6cc7-4f7a-ab64-4a34b12bee43.png">
<img width="550" alt="Screen Shot 2022-05-17 at 4 09 41 PM" src="https://user-images.githubusercontent.com/28973352/168910680-4c204b75-c417-4b64-b6e4-2e35e7f2b3d0.png">
</p>


## UI Screenshots
<img width="639" alt="Screen Shot 2022-05-14 at 11 02 15 AM" src="https://user-images.githubusercontent.com/28973352/168439644-355a1367-dd28-43e3-8527-b2d2b903bfde.png">
In the above, color coding helps to understand the level of severity associated with the disease. Darker the color, severe is the disease.

## Evaluation Results
<img width="623" alt="Screen Shot 2022-05-14 at 11 03 00 AM" src="https://user-images.githubusercontent.com/28973352/168439704-457ae4c1-b401-4ec5-a0b2-1bdc8aa93e87.png">
