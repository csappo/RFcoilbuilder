# coilbuilding
Tools under development for coil design and building in the low field lab at VUIIS

## Simulation 
```python coin_optimisation_v1.3_DIYMRI.py ```

This will generate png images for B1 field and winding pattern. You may want to tune parameters within this file to generate a uniform B1 field. 

<img src="https://github.com/csappo/RFcoilbuilder/blob/main/01_field_performance.png">

<img src="https://github.com/csappo/RFcoilbuilder/blob/main/03_coil_developed_view.png">


## STL creation for 3D printing
The above code with generate a csv file, run the following to generate STL file

```python generate_former_stl.py```

update path to ```default_csv``` in the file


