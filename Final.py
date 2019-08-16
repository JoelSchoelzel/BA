import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io
from tkinter import filedialog
from tkinter import *
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
import matplotlib.font_manager as font_manager
import seaborn as sns

font_dirs = ['Schriftarten']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Heuristica'
plt.rcParams['mathtext.it'] = 'Heuristica:bold'
plt.rcParams['mathtext.bf'] = 'Heuristica:bold'
plt.rcParams['font.family'] = 'Heuristica'
plt.rcParams['figure.figsize'] = [7.5, 5.6]
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=14)  # fontsize of the figure title

##Einlesen der Daten
root =Tk()
filename =  filedialog.askopenfilename(title = 'Wähle Anteil PV', filetypes = (("mat files","*.mat"),("all files","*.*")))
filename2 =  filedialog.askopenfilename(title = 'Öffne Datei mit Erzeugung durch BHKW', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename2 == '':
    filename2 = 'txt_zeros.txt'
filename3 =  filedialog.askopenfilename(title = 'Öffne Datei für EV-Batterien', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename3 == '':
    filename3 = 'txt_zeros.txt'
filename4 =  filedialog.askopenfilename(title = 'Öffne Datei für Speicherbatterien', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename4 == '':
    filename4 = 'txt_zeros.txt'
filename5 =  filedialog.askopenfilename(title = 'Öffne Datei powerHP', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename5 == '':
    filename5 = 'txt_zeros.txt'
filename6 =  filedialog.askopenfilename(title = 'Öffne Datei powerHR', filetypes = (("txt files","*.txt"),("all files","*.*")))
if filename6 == '':
    filename6 = 'txt_zeros.txt'


##Erzeugung durch PV
new_dict = dict()
#mat = scipy.io.loadmat('Daten/profiles_LV_suburban_PV_25', mdict= new_dict)
mat = scipy.io.loadmat(filename, mdict= new_dict)
var1 = mat.get('p_gen')
var1 = var1/1000
l = len(var1)
l_35040 = len(var1)

array_p_gen2 = np.array([])   #Summe aller p_gen
for i in range(l):
	x = var1[i]
	y = (np.sum(x))
	array_p_gen2 = np.append(array_p_gen2, y)

array_p_gen = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_p_gen2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_p_gen = np.append(array_p_gen, mean_4)


##Verbrauch durch Haushalte
new_dict_2 = dict()
#mat = scipy.io.loadmat('Daten/profiles_LV_suburban_PV_25', mdict= new_dict_2)
mat = scipy.io.loadmat(filename, mdict= new_dict_2)
var2 = mat.get('p_dem')
var2 = var2/1000

array_p_dem2 = np.array([])   #Summe aller p_dem
for i in range(l):
	x = var2[i]
	y = (np.sum(x))
	array_p_dem2 = np.append(array_p_dem2, y)

array_p_dem = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_p_dem2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_p_dem = np.append(array_p_dem, mean_4)

##Erzeugung durch BHKW
#matrix_BHKW_gen = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerCHP.txt', skiprows=0)
matrix_BHKW_gen = np.loadtxt(filename2, skiprows=0)
matrix_BHKW_gen = matrix_BHKW_gen[:35040, :]
array_BHKW_gen2 = matrix_BHKW_gen.sum(axis=1)

array_BHKW_gen = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_BHKW_gen2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_BHKW_gen = np.append(array_BHKW_gen, mean_4)

##Einfluss der EV-Batterien
#matrix_EV = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerEV.txt', skiprows=0)
matrix_EV = np.loadtxt(filename3, skiprows=0)
matrix_EV = matrix_EV[:35040, :]
array_EV2 = matrix_EV.sum(axis=1)

array_EV = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_EV2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_EV = np.append(array_EV, mean_4)

##Einfluss der Speicherbatterien
#matrix_Bat = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerBat.txt', skiprows=0)
matrix_Bat = np.loadtxt(filename4, skiprows=0)
matrix_Bat = matrix_Bat[:35040, :]
array_Bat2 = matrix_Bat.sum(axis=1)

array_Bat = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_Bat2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_Bat = np.append(array_Bat, mean_4)
len_Bat = len(array_Bat)

##Lade- und Endladevorgänge voneinander trennen und in 2 arrays packen
array_Bat_einspeisen = np.array([])
array_Bat_speichern = np.array([])
for i in range(len_Bat):
    if array_Bat[i] > 0:
        array_Bat_speichern = np.append(array_Bat_speichern, array_Bat[i])
        array_Bat_einspeisen = np.append(array_Bat_einspeisen, 0)
    elif array_Bat[i] <= 0:
        array_Bat_speichern = np.append(array_Bat_speichern, 0)
        array_Bat_einspeisen = np.append(array_Bat_einspeisen, array_Bat[i])

##Stromverbrauch durch HP
#matrix_HP = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerHP.txt', skiprows=0)
matrix_HP = np.loadtxt(filename5, skiprows=0)
matrix_HP = matrix_HP[:35040, :]
array_HP2 = matrix_HP.sum(axis=1)

array_HP = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_HP2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_HP = np.append(array_HP, mean_4)

##Stromverbrauch durch HR
#matrix_HR = np.loadtxt('Daten/HP75_CHP25_EV25_BAT25_TES100_CO2mix/powerHR.txt', skiprows=0)
matrix_HR = np.loadtxt(filename6, skiprows=0)
matrix_HR = matrix_HR[:35040, :]
array_HR2 = matrix_HR.sum(axis=1)

array_HR = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_HR2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_HR = np.append(array_HR, mean_4)


##Wärmebedarf Trinkwarmwasser
matrix_QDHW = np.loadtxt('Daten/dotQDHW.txt', skiprows=0)
#matrix_QDHW = np.loadtxt(filenameeeeeee, skiprows=1)
array_QDHW2 = matrix_QDHW.sum(axis=1)

array_QDHW = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_QDHW2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_QDHW = np.append(array_QDHW, mean_4)

##Wärmebedarf Heizen
matrix_SH = np.loadtxt('Daten/dotQSH.txt', skiprows=0)
#matrix_SH = np.loadtxt(filenameeeeeeee, skiprows=1)
array_SH2 = matrix_SH.sum(axis=1)

array_SH = np.array([])
for i in range(0, l, 4):
    steps_mean = np.array([])
    for j in range(4):
        one = array_SH2[i + j]
        steps_mean = np.append(steps_mean, one)
    mean_4 = np.mean(steps_mean)
    array_SH = np.append(array_SH, mean_4)

### matrizen in 1-Stunden-Takt ändern ###
matrix_HP2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_HP[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_HP2 = np.append(matrix_HP2, array2, axis=0)
matrix_HP2 = np.reshape(matrix_HP2, (120, 8760))
matrix_HP2 = matrix_HP2.T

matrix_HR2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_HR[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_HR2 = np.append(matrix_HR2, array2, axis=0)
matrix_HR2 = np.reshape(matrix_HR2, (120, 8760))
matrix_HR2 = matrix_HR2.T

matrix_Bat2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_Bat[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_Bat2 = np.append(matrix_Bat2, array2, axis=0)
matrix_Bat2 = np.reshape(matrix_Bat2, (120, 8760))
matrix_Bat2 = matrix_Bat2.T

matrix_EV2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_EV[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_EV2 = np.append(matrix_EV2, array2, axis=0)
matrix_EV2 = np.reshape(matrix_EV2, (120, 8760))
matrix_EV2 = matrix_EV2.T

matrix_BHKW_gen2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_BHKW_gen[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_BHKW_gen2 = np.append(matrix_BHKW_gen2, array2, axis=0)
matrix_BHKW_gen2 = np.reshape(matrix_BHKW_gen2, (120, 8760))
matrix_BHKW_gen2 = matrix_BHKW_gen2.T

var12 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = var1[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    var12 = np.append(var12, array2, axis=0)
var12 = np.reshape(var12, (120, 8760))
var12 = var12.T

var22 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = var2[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    var22 = np.append(var22, array2, axis=0)
var22 = np.reshape(var22, (120, 8760))
var22 = var22.T

matrix_SH2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_SH[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_SH2 = np.append(matrix_SH2, array2, axis=0)
matrix_SH2 = np.reshape(matrix_SH2, (120, 8760))
matrix_SH2 = matrix_SH2.T

matrix_QDHW2 = np.array([])
for i in range(120):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = matrix_QDHW[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    matrix_QDHW2 = np.append(matrix_QDHW2, array2, axis=0)
matrix_QDHW2 = np.reshape(matrix_QDHW2, (120, 8760))
matrix_QDHW2 = matrix_QDHW2.T


l = len(array_SH)
number_of_months = int(12)
number_of_days = int(365)
number_of_weeks = int(52)
len_day = l / number_of_days
len_day = int(len_day)
len_week = 168
len_week = int(len_week)
len_month = l / number_of_months
len_month = int(len_month)

##Kabeltypauswahl für Bemessungsstrom
Bemessungsstrom_Kabeltypen = pd.read_csv("Strombelastbarkeit_Kabel_in_Erde.csv", skiprows=2, usecols=range(1,7), header=None, sep=";", engine='python')

##CO2-äquivalente Emissionen
erzeugung2 = pd.read_csv("TransnetBW_Actual generation_201801010000_201812312345_1.csv", skiprows=1, usecols=range(2,14), header=None, sep=";")
#erzeugung2 = pd.read_csv(filenameeeeeeeee, skiprows=1, usecols=range(2,14), header=None, sep=";")

erzeugung = np.array([])
for i in range(0, 12, 1):
    array2 = np.array([])
    for j in range(0, 35040, 4):
        array1 = np.array([])
        for q in range(4):
            a = erzeugung2.iat[j + q , i]
            array1 = np.append(array1, a)
        mean = np.mean(array1)
        array2 = np.append(array2, mean)
    erzeugung = np.append(erzeugung, array2, axis=0)
erzeugung = np.reshape(erzeugung, (12, 8760))
erzeugung = erzeugung.T

ABC = pd.read_csv("CO2_Emissionsfaktoren.csv", skiprows=1, usecols=range(0,12), header=None, sep=";")
#ABC = pd.read_csv(filenameeeeeeeeee, skiprows=1, usecols=range(0,12), header=None, sep=";")

cos_phi = 0.95 #für Umrechnung von Wirk zu Scheinleistung

###Engpässe
##Engpassgefahr##

#Grundlast_min = np.min(var22)
#Engpassgefahr_Last = 630 - (AnzahlHP*LeistungHP)/cos_phi - Grundlast_min/cos_phi
#Engpassgefahr_Erzeugung = 630 - (AnzahlPV*LeistungPV)/cos_phi - (AnzahlBHKW*LeistungBHKW)/cos_phi + Grundlast_min/cos_phi


##Engpässe an den Hausanschlüssen
Leistung_Hausanschluss = var22 + matrix_EV2 + matrix_Bat2 + matrix_HP2 + matrix_HR2 + var12 - matrix_BHKW_gen2

Stromsicherungen = pd.read_csv("SicherungenHausanschlüsse.csv", skiprows=1, usecols=range(0,120), header=None, sep=";", engine='python')

Scheinleistung_Hausanschluss = Leistung_Hausanschluss / cos_phi

#matrix_Scheinleistung_Hausanschluss_pos = np.array([])
#matrix_Scheinleistung_Hausanschluss_neg = np.array([])
#for i in range(l):
#    array_pos = np.array([])
#    array_neg = np.array([])
#    for j in range(120):
#        if Scheinleistung_Hausanschluss[i, j] > 0:
#            array_pos = np.append(array_pos, Scheinleistung_Hausanschluss[i, j])
#            array_neg = np.append(array_neg, 0)
#        elif Scheinleistung_Hausanschluss[i, j] < 0:
#            array_pos = np.append(array_pos, 0)
#            array_neg = np.append(array_neg, Scheinleistung_Hausanschluss[i, j])
#    matrix_Scheinleistung_Hausanschluss_pos = np.append(matrix_Scheinleistung_Hausanschluss_pos, array_pos, axis=0)
#    matrix_Scheinleistung_Hausanschluss_neg = np.append(matrix_Scheinleistung_Hausanschluss_neg, array_neg, axis=0)


Bemessungsleistung_Hausanschluss = Stromsicherungen *3 *230 /1000       #/1000 -> kWh
array_Bemessungsleistung_Hausanschluss = Bemessungsleistung_Hausanschluss.values     #120 Häuser -> 120 Spalten

##Engpässe in den Kabeln
buildingsFeeder = np.loadtxt('Daten/buildingFeeder.txt')
unique, counts = np.unique(buildingsFeeder, return_counts=True)    #counts gibt Anzahl der Haushalte an der Feeder
dict(zip(unique, counts))                                          #unique gibt Nummerierung der Feeder an

u = counts[0]
v = counts[1]
w = counts[2]
x = counts[3]
y = counts[4]
z = counts[5]

feeder1 = Scheinleistung_Hausanschluss[:, :u].sum(axis=1)         #Wirkleistungen an den Sammelschienen
feeder2 = Scheinleistung_Hausanschluss[:, u:(u+v)].sum(axis=1)
feeder3 = Scheinleistung_Hausanschluss[:, (u+v):(u+v+w)].sum(axis=1)
feeder4 = Scheinleistung_Hausanschluss[:, (u+v+w):(u+v+w+x)].sum(axis=1)
feeder5 = Scheinleistung_Hausanschluss[:, (u+v+w+x):(u+v+w+x+y)].sum(axis=1)
feeder6 = Scheinleistung_Hausanschluss[:, (u+v+w+x+y):].sum(axis=1)

Bemessungsleistung_feeder1 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V    /1000 kVA
Bemessungsleistung_feeder2 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V    #Kabelauswahl nach Kerber Vorstadtnetz
Bemessungsleistung_feeder3 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder4 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder5 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V
Bemessungsleistung_feeder6 = 144 * 230 /1000    #Bemessungsstrom: 133 A, Nennspannung: 230 V

array_Netzengpassleistung_feeder1_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder1_pos = np.array([])
array_Netzengpassleistung_feeder1_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder1_neg = np.array([])
for i in range(l):
    if feeder1[i] > 0:
        if feeder1[i] - Bemessungsleistung_feeder1 > 0:
            array_Netzengpassleistung_feeder1_pos = np.append(array_Netzengpassleistung_feeder1_pos, (feeder1[i] - Bemessungsleistung_feeder1))
        elif Bemessungsleistung_feeder1 - feeder1[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder1_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder1_pos, (Bemessungsleistung_feeder1 - feeder1[i]))
    if feeder1[i] < 0:
        if feeder1[i] + Bemessungsleistung_feeder1 < 0:
            array_Netzengpassleistung_feeder1_neg = np.append(array_Netzengpassleistung_feeder1_neg, (-1) * (feeder1[i] + Bemessungsleistung_feeder1))
        elif Bemessungsleistung_feeder1 + feeder1[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder1_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder1_neg, (Bemessungsleistung_feeder1 + feeder1[i]))

array_Netzengpassleistung_feeder2_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder2_pos = np.array([])
array_Netzengpassleistung_feeder2_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder2_neg = np.array([])
for i in range(l):
    if feeder2[i] > 0:
        if feeder2[i] - Bemessungsleistung_feeder2 > 0:
            array_Netzengpassleistung_feeder2_pos = np.append(array_Netzengpassleistung_feeder2_pos, (feeder2[i] - Bemessungsleistung_feeder2))
        elif Bemessungsleistung_feeder2 - feeder2[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder2_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder2_pos, (Bemessungsleistung_feeder2 - feeder2[i]))
    if feeder2[i] < 0:
        if feeder2[i] + Bemessungsleistung_feeder2 < 0:
            array_Netzengpassleistung_feeder2_neg = np.append(array_Netzengpassleistung_feeder2_neg, (-1) * (feeder2[i] + Bemessungsleistung_feeder2))
        elif Bemessungsleistung_feeder2 + feeder2[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder2_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder2_neg, (Bemessungsleistung_feeder2 + feeder2[i]))

array_Netzengpassleistung_feeder3_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder3_pos = np.array([])
array_Netzengpassleistung_feeder3_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder3_neg = np.array([])
for i in range(l):
    if feeder3[i] > 0:
        if feeder3[i] - Bemessungsleistung_feeder3 > 0:
            array_Netzengpassleistung_feeder3_pos = np.append(array_Netzengpassleistung_feeder3_pos, (feeder3[i] - Bemessungsleistung_feeder3))
        elif Bemessungsleistung_feeder3 - feeder3[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder3_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder3_pos, (Bemessungsleistung_feeder3 - feeder3[i]))
    if feeder3[i] < 0:
        if feeder3[i] + Bemessungsleistung_feeder3 < 0:
            array_Netzengpassleistung_feeder3_neg = np.append(array_Netzengpassleistung_feeder3_neg, (-1) * (feeder3[i] + Bemessungsleistung_feeder3))
        elif Bemessungsleistung_feeder3 + feeder3[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder3_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder3_neg, (Bemessungsleistung_feeder3 + feeder3[i]))

array_Netzengpassleistung_feeder4_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder4_pos = np.array([])
array_Netzengpassleistung_feeder4_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder4_neg = np.array([])
for i in range(l):
    if feeder4[i] > 0:
        if feeder4[i] - Bemessungsleistung_feeder4 > 0:
            array_Netzengpassleistung_feeder4_pos = np.append(array_Netzengpassleistung_feeder4_pos, (feeder4[i] - Bemessungsleistung_feeder4))
        elif Bemessungsleistung_feeder4 - feeder4[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder4_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder4_pos, (Bemessungsleistung_feeder4 - feeder4[i]))
    if feeder4[i] < 0:
        if feeder4[i] + Bemessungsleistung_feeder4 < 0:
            array_Netzengpassleistung_feeder4_neg = np.append(array_Netzengpassleistung_feeder4_neg, (-1) * (feeder4[i] + Bemessungsleistung_feeder4))
        elif Bemessungsleistung_feeder4 + feeder4[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder4_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder4_neg, (Bemessungsleistung_feeder4 + feeder4[i]))

array_Netzengpassleistung_feeder5_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder5_pos = np.array([])
array_Netzengpassleistung_feeder5_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder5_neg = np.array([])
for i in range(l):
    if feeder5[i] > 0:
        if feeder5[i] - Bemessungsleistung_feeder5 > 0:
            array_Netzengpassleistung_feeder5_pos = np.append(array_Netzengpassleistung_feeder5_pos, (feeder5[i] - Bemessungsleistung_feeder5))
        elif Bemessungsleistung_feeder5 - feeder5[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder5_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder5_pos, (Bemessungsleistung_feeder5 - feeder5[i]))
    if feeder5[i] < 0:
        if feeder5[i] + Bemessungsleistung_feeder5 < 0:
            array_Netzengpassleistung_feeder5_neg = np.append(array_Netzengpassleistung_feeder5_neg, (-1) * (feeder5[i] + Bemessungsleistung_feeder5))
        elif Bemessungsleistung_feeder5 + feeder5[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder5_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder5_neg, (Bemessungsleistung_feeder5 + feeder5[i]))

array_Netzengpassleistung_feeder6_pos = np.array([])
array_Leistungsaufnahmefähigkeit_feeder6_pos = np.array([])
array_Netzengpassleistung_feeder6_neg = np.array([])
array_Leistungsaufnahmefähigkeit_feeder6_neg = np.array([])
for i in range(l):
    if feeder6[i] > 0:
        if feeder6[i] - Bemessungsleistung_feeder6 > 0:
            array_Netzengpassleistung_feeder6_pos = np.append(array_Netzengpassleistung_feeder6_pos, (feeder6[i] - Bemessungsleistung_feeder6))
        elif Bemessungsleistung_feeder6 - feeder6[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder6_pos = np.append(array_Leistungsaufnahmefähigkeit_feeder6_pos, (Bemessungsleistung_feeder6 - feeder6[i]))
    if feeder6[i] < 0:
        if feeder6[i] + Bemessungsleistung_feeder6 < 0:
            array_Netzengpassleistung_feeder6_neg = np.append(array_Netzengpassleistung_feeder6_neg, (-1) * (feeder6[i] + Bemessungsleistung_feeder6))
        elif Bemessungsleistung_feeder6 + feeder6[i] > 0:
            array_Leistungsaufnahmefähigkeit_feeder6_neg = np.append(array_Leistungsaufnahmefähigkeit_feeder6_neg, (Bemessungsleistung_feeder6 + feeder6[i]))

plt.hist(array_Leistungsaufnahmefähigkeit_feeder1_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder1_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 1')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 1.png')
plt.ylabel('Häufigkeit')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_feeder2_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder2_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 2')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 2.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_feeder3_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder3_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 3')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 3.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_feeder4_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder4_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 4')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 4.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_feeder5_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder5_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 5')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 5.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_feeder6_pos, color='gray', bins=75, alpha=0.7)
plt.hist(array_Leistungsaufnahmefähigkeit_feeder6_neg, color='red', bins=75, alpha=0.5)
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistung-\nsaufnahmefähigkeit von Strang 6')
plt.ylabel('Häufigkeit')
plt.xlabel('Leistung [kW]')
plt.xlim((0,150))
plt.ylim((0,250))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 6.png')
plt.show()


##Engpässe an der Ortnetzstation
Bemessungsleistung_Ortsnetzstation = 630                 # kVA maximale Transformatorleistung
Last_Ortsnetzstation = Scheinleistung_Hausanschluss.sum(axis=1)

array_Netzengpassleistung_Ortsnetzstation_pos = np.array([])
array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.array([])
array_Netzengpassleistung_Ortsnetzstation_neg = np.array([])
array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.array([])
for i in range(l):
    if Last_Ortsnetzstation[i] > 0:
        if Last_Ortsnetzstation[i] - Bemessungsleistung_Ortsnetzstation > 0:
            array_Netzengpassleistung_Ortsnetzstation_pos = np.append(array_Netzengpassleistung_Ortsnetzstation_pos, (Last_Ortsnetzstation[i] - Bemessungsleistung_Ortsnetzstation))
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, 0)
        elif Bemessungsleistung_Ortsnetzstation - Last_Ortsnetzstation[i] > 0:
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, (Bemessungsleistung_Ortsnetzstation - Last_Ortsnetzstation[i]))
    elif Last_Ortsnetzstation[i] < 0:
        if Last_Ortsnetzstation[i] + Bemessungsleistung_Ortsnetzstation < 0:
            array_Netzengpassleistung_Ortsnetzstation_neg = np.append(array_Netzengpassleistung_Ortsnetzstation_neg, (-1) * (Last_Ortsnetzstation[i] + Bemessungsleistung_Ortsnetzstation))
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, 0)
        elif Bemessungsleistung_Ortsnetzstation + Last_Ortsnetzstation[i] > 0:
            array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg = np.append(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, (Bemessungsleistung_Ortsnetzstation + Last_Ortsnetzstation[i]))


plt.hist(array_Netzengpassleistung_Ortsnetzstation_pos, color='dimgray', bins=60, range=[0, 300])
plt.hist(array_Netzengpassleistung_Ortsnetzstation_neg, color='red', bins=60, alpha=0.5, range=[0, 300])
#plt.grid(axis='y')
#plt.title('Häufigkeitsverteilung der Engpassleistung an der Ortsnetzstation')
plt.xlabel('Leistung [kVA]')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.ylim((0, 25))
plt.xlim((0, 300))
plt.savefig('Häufigkeitsverteilung_der_Engpassleistung_Ortsnetzstation.png')
plt.show()

plt.hist(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_pos, color='dimgray', bins=140, range=[0, 700], label='Strombezug')
plt.hist(array_Leistungsaufnahmefähigkeit_Ortsnetzstation_neg, color='red', bins=140, alpha=0.5, range=[0, 700], label='Stromeinspeisung')
#plt.grid(axis='y')
plt.title('Häufigkeitsverteilung der Leistungs-\naufnahmefähigkeit an der Ortsnetzstation')
plt.legend()
plt.xlabel('Leistung [kVA]')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.ylim((0,400))
plt.xlim((0, 700))
plt.savefig('Häufigkeitsverteilung_der_Leistungsaufnahmefähigkeit_Ortsnetzstation.png')
plt.show()

##Engpassarbeit

Engpassarbeit_neg = np.sum(array_Netzengpassleistung_Ortsnetzstation_neg)   #kWh
print('Die Engpassarbeit an der Ortnetzstation, die durch Erzeugung entsteht, beträgt ' + str(Engpassarbeit_neg) + ' kWh')
Engpassarbeit_pos = np.sum(array_Netzengpassleistung_Ortsnetzstation_pos)      #kWh
print('Die Engpassarbeit an der Ortnetzstation, die durch Last entsteht, beträgt ' + str(Engpassarbeit_pos) + ' kWh')



###   GSC mit EE-Erzeugung im Quartier   ###

#array_p_gen_mean_d = np.array([]) #durchschn erzeugte Leistung pro tag
#for i in range(0, l, len_day):
#    P_GEN = np.array([])
#    for j in range(len_day):
#        b = np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])
#        P_GEN = np.append(P_GEN, b)
#    mean_d = np.mean(P_GEN)
#    array_p_gen_mean_d = np.append(array_p_gen_mean_d, mean_d)    #mittelwert erzeugte leistung pro tag
#
#array_p_gen_all = np.array([])
#for i in range(l):
#    a = np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])
#    array_p_gen_all = np.append(array_p_gen_all, a)
#
#array_p_dem_d = np.array([])  #Verbrauch pro tag
#for i in range(0, l, len_day):
#    P_DEM = np.array([])
#    for j in range(len_day):
#        c = array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]
#        P_DEM = np.append(P_DEM, c)
#
#    sum_d = np.sum(P_DEM)
#    array_p_dem_d = np.append(array_p_dem_d, sum_d)  # Verbrauch pro tag für 365 tage
#
#
#array_GSC_abs = np.array([])
#for jj in range(0, l, len_day):
#    i = 0
#    array_dem_mal_gen = np.array([])
#    for ii in range(len_day):
#        yy = (array_p_dem[ii + jj] + array_HP[ii + jj] + array_HR[ii + jj] + array_EV[ii + jj] + array_Bat_speichern[ii + jj]) * array_p_gen_all[ii + jj]
#        array_dem_mal_gen = np.append(array_dem_mal_gen, yy)
#
#    sum_array_dem_mal_gen = np.sum(array_dem_mal_gen)
#    GSC_abs = (sum_array_dem_mal_gen) / (array_p_dem_d[i] * array_p_gen_mean_d[i])
#    array_GSC_abs = np.append(array_GSC_abs, GSC_abs)
#    i = i + 1
#
#plt.hist(array_GSC_abs, bins=50)
#plt.title('Häufigkeitsverteilung GSC_abs')
#plt.ylabel('Menge an GSC_abs')
#plt.xlabel('GSC_abs')
#plt.xlim((0,5))
#plt.xticks(([0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5, 4, 4.5, 5]))
#plt.ylim((0, 65))
#plt.savefig('GSC_abs.png')
#plt.show()
#
#
#
#W_el_d_array = np.array([])
#for j in range(0, l, len_day):
#    W_el_d = np.array([])
#    for i in range(len_day):
#        y = array_p_dem[j + i] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]                    #alle Stromverbraucher!!!
#        W_el_d = np.append(W_el_d, y)       #
#
#    sum_W_el_d = np.sum(W_el_d)
#    W_el_d_array = np.append(W_el_d_array, sum_W_el_d)
#
#
#
#h_fl = W_el_d_array / ((120*4.8) + (0*10))   #Anzahl und Leistung der Erzeuger(PV und BHKW)   #BHKW 10 weil 9, höchste in txt datei
#
#h_fl_round = np.around(h_fl)
#h_fl_round2 = h_fl_round
#h_fl_round2 = h_fl_round2.astype(int)
#sum_h_fl_round2 = np.sum(h_fl_round2)
#
#h = -1
#array_pos_min = np.array([])
#array_pos_max = np.array([])
#for j in range(0, l, len_day):
#    h = h + 1#
#    x = h_fl_round2[h]
#    x = int(x)
#
#    array_gen_best_worst2 = np.array([])
#    for i in range(len_day):
#            z = np.absolute(array_p_gen[j + i]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])
#            array_gen_best_worst2 = np.append(array_gen_best_worst2, z)
#    array_gen_best_worst2_sorted = np.sort(array_gen_best_worst2)
#    array_gen_best_worst2_sorted = np.unique(array_gen_best_worst2_sorted)
#
#    pos_min_values = np.array([])
#    for q in range(x):
#        pos_min = np.where(array_gen_best_worst2 == array_gen_best_worst2_sorted[q])
#        pos_min = np.asarray(pos_min)
#        pos_min_values = np.append(pos_min_values, pos_min[0][:h_fl_round2[h]] + h * len_day)
#        if len(pos_min_values) >= h_fl_round2[h]:
#            pos_min_values = pos_min_values[:h_fl_round2[h]]
#            break
#    array_pos_min = np.append(array_pos_min, pos_min_values)
#
#    pos_max_values = np.array([])
#    for q in range(x):
#        pos_max = np.where(array_gen_best_worst2 == array_gen_best_worst2_sorted[-q - 1])
#
#        pos_max = np.asarray(pos_max)
#        pos_max_values = np.append(pos_max_values, pos_max[0][:h_fl_round2[h]] + h * len_day)
#        if len(pos_max_values) >= h_fl_round2[h]:
#            pos_max_values = pos_max_values[:h_fl_round2[h]]
#            break
#    array_pos_max = np.append(array_pos_max, pos_max_values)
#
#max_load_per_step = W_el_d_array / h_fl_round2      # Watt
#max_load_per_step = max_load_per_step.astype(int)
#
#
#Last_verlegt_max = np.zeros(l, dtype=int)
#Last_verlegt_min = np.zeros(l, dtype=int)
#
#m = 0
#for i in range(365):
#    for j in range(h_fl_round2[i]):
#        array_max = np.put(Last_verlegt_max, (array_pos_max[m + j]), max_load_per_step[i])
#    m = m + h_fl_round2[i]
#
#m = 0
#for i in range(365):
#    for j in range(h_fl_round2[i]):
#        array_min = np.put(Last_verlegt_min, (array_pos_min[m + j]), max_load_per_step[i])
#    m = m + h_fl_round2[i]
#
#
#array_p_dem_d_max = np.array([])  #Verbrauch pro tag
#for i in range(0, l, len_day):
#    P_DEM_max = np.array([])
#    for j in range(len_day):
#        c = Last_verlegt_max[i + j]
#        P_DEM_max = np.append(P_DEM_max, c)
#
#    sum_d = np.sum(P_DEM_max)
#    array_p_dem_d_max = np.append(array_p_dem_d_max, sum_d)  # Verbrauch pro tag für 365 tage
#
#array_GSC_abs_max = np.array([])
#for j in range(0, l, len_day):
#    läufer = 0
#    array_dem_mal_gen_max = np.array([])
#    for i in range(len_day):
#        yy = Last_verlegt_max[i + j] * array_p_gen_all[i + j]
#        array_dem_mal_gen_max = np.append(array_dem_mal_gen_max, yy)
#
#    sum_array_dem_mal_gen_max = np.sum(array_dem_mal_gen_max)
#    GSC_abs_max = (sum_array_dem_mal_gen_max) / (array_p_dem_d_max[läufer] * array_p_gen_mean_d[läufer])
#    array_GSC_abs_max = np.append(array_GSC_abs_max, GSC_abs_max)
#    läufer = läufer + 1
#
#array_GSC_abs_min = np.array([])
#for j in range(0, l, len_day):
#    läufer = 0
#    array_dem_mal_gen_min = np.array([])
#    for i in range(len_day):
#        yy = Last_verlegt_min[i + j] * array_p_gen_all[i + j]
#        array_dem_mal_gen_min = np.append(array_dem_mal_gen_min, yy)
#
#    sum_array_dem_mal_gen_min = np.sum(array_dem_mal_gen_min)
#    GSC_abs_min = (sum_array_dem_mal_gen_min) / (array_p_dem_d_max[läufer] * array_p_gen_mean_d[läufer])
#    array_GSC_abs_min = np.append(array_GSC_abs_min, GSC_abs_min)
#    läufer = läufer + 1
#
#
#GSC_rel = 200 *((array_GSC_abs_min - array_GSC_abs) / (array_GSC_abs_min - array_GSC_abs_max)) - 100
#
#GSC_rel_max = np.max(GSC_rel)
#GSC_rel_min = np.min(GSC_rel)
#
#print('Der maximale Tageswert von GSC_rel beträgt' + str(GSC_rel_max))
#print('Der minimale Tageswert von GSC_rel beträgt' + str(GSC_rel_min))
#
GSC_rel = np.array([])

plt.hist(GSC_rel, bins=100, color='red', range=[-100, 100])
#plt.title('Häufigkeitsverteilung des relativen GSC')
plt.xlim((-100,100))
plt.xticks([-100, -50, 0, 50, 100])
plt.ylim((0, 24))
plt.ylabel('Anzahl täglicher Werte')
plt.savefig('GSC_rel.png')
plt.show()

###   Deckungsgrad   ###

##pro tag
list1 = []
for i in range(0, l, len_day):
    p_dem_96 = np.array([])
    for j in range(len_day):
        cc = array_p_dem[i + j]
        p_dem_96 = np.append(p_dem_96, cc)
    sum1 = np.sum(p_dem_96)
    list1.append(sum1)
p_dem_ndarray = np.asarray(list1)

list2 = []
for i in range(0, l, len_day):
    p_gen_96 = np.array([])
    for j in range(len_day):
        cc = array_p_gen[i + j]
        p_gen_96 = np.append(p_gen_96, cc)
    sum2 = np.sum(p_gen_96)
    list2.append(sum2)
p_gen_ndarray = np.asarray(list2)

list3 = []
for i in range(0, l, len_day):
    BHKW_gen_96 = np.array([])
    for j in range(len_day):
        cc = array_BHKW_gen[i + j]
        BHKW_gen_96 = np.append(BHKW_gen_96, cc)
    sum3 = np.sum(BHKW_gen_96)
    list3.append(sum3)
BHKW_gen_ndarray = np.asarray(list3)

list4 = []
for i in range(0, l, len_day):
    EV_96 = np.array([])
    for j in range(len_day):
        cc = array_EV[i + j]
        EV_96 = np.append(EV_96, cc)
    sum4 = np.sum(EV_96)
    list4.append(sum4)
EV_ndarray = np.asarray(list4)

list5 = []
for i in range(0, l, len_day):
    Bat_einspeisen_96 = np.array([])
    for j in range(len_day):
        cc = array_Bat_einspeisen[i + j]
        Bat_einspeisen_96 = np.append(Bat_einspeisen_96, cc)
    sum5 = np.sum(Bat_einspeisen_96)
    list5.append(sum5)
Bat_einspeisen_ndarray = np.asarray(list5)

list6 = []
for i in range(0, l, len_day):
    Bat_speichern_96 = np.array([])
    for j in range(len_day):
        cc = array_Bat_speichern[i + j]
        Bat_speichern_96 = np.append(Bat_speichern_96, cc)
    sum6 = np.sum(Bat_speichern_96)
    list6.append(sum6)
Bat_speichern_ndarray = np.asarray(list6)

list7 = []
for i in range(0, l, len_day):
    HP_96 = np.array([])
    for j in range(len_day):
        cc = array_HP[i + j]
        HP_96 = np.append(HP_96, cc)
    sum7 = np.sum(HP_96)
    list7.append(sum7)
HP_ndarray = np.asarray(list7)

list8 = []
for i in range(0, l, len_day):
    HR_96 = np.array([])
    for j in range(len_day):
        cc = array_HR[i + j]
        HR_96 = np.append(HR_96, cc)
    sum8 = np.sum(HR_96)
    list8.append(sum8)
HR_ndarray = np.asarray(list8)
ll = len(HR_ndarray)

#Deckungsgrad
array_gamma_DG = np.array([])
for i in range(ll):

    if (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ) \
            > ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) ):
        gamma_DG = (( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) )
                    / (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] )) * 100
    elif ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]) ) \
            > (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ):
        gamma_DG = 100
    array_gamma_DG = np.append(array_gamma_DG, gamma_DG)

DG_min_monat = np.array([])
DG_max_monat = np.array([])
for i in range(0, number_of_days-5, 30):
    DG_min_max = np.array([])
    for j in range(30):
        cc = array_gamma_DG[i + j]
        DG_min_max = np.append(DG_min_max, cc)
    DG_min = np.min(DG_min_max)
    DG_max = np.max(DG_min_max)
    DG_min_monat = np.append(DG_min_monat, DG_min)
    DG_max_monat = np.append(DG_max_monat, DG_max)

DG_min_monat = DG_min_monat.astype(int)
DG_max_monat = DG_max_monat.astype(int)
print('Die minimalen täglichen Deckungsgrade der Monate' + str(DG_min_monat))
print('Die maximalen täglichen Deckungsgrade der Monate' + str(DG_max_monat))

#Eigenverbrauch
array_gamma_EV = np.array([])
for i in range(ll):

    if ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])) \
            > (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ):
        gamma_EV = ((p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] )
                    / (np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i]))) *100
    elif ( np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])) == 0:
        gamma_EV = 0
    elif (p_dem_ndarray[i] + HP_ndarray[i] + HR_ndarray[i] + EV_ndarray[i] + Bat_speichern_ndarray[i] ) \
            > (  np.absolute(p_gen_ndarray[i]) + BHKW_gen_ndarray[i] + np.absolute(Bat_einspeisen_ndarray[i])):
        gamma_EV = 100
    array_gamma_EV = np.append(array_gamma_EV, gamma_EV)

EV_min_monat = np.array([])
EV_max_monat = np.array([])
for i in range(0, number_of_days-5, 30):
    EV_min_max = np.array([])
    for j in range(30):
        cc = array_gamma_EV[i + j]
        EV_min_max = np.append(EV_min_max, cc)
    EV_min = np.min(EV_min_max)
    EV_max = np.max(EV_min_max)
    EV_min_monat = np.append(EV_min_monat, EV_min)
    EV_max_monat = np.append(EV_max_monat, EV_max)

EV_min_monat = EV_min_monat.astype(int)
EV_max_monat = EV_max_monat.astype(int)
print('Die minimalen täglichen Deckungsgrade der Monate' + str(EV_min_monat))
print('Die maximalen täglichen Deckungsgrade der Monate' + str(EV_max_monat))


#Autarkie
netto_stromlast = (array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern) \
                  - np.absolute(array_p_gen) - array_BHKW_gen - np.absolute(array_Bat_einspeisen)

#Autarkie
array_Autarkie_15_min = np.array([])
for q in range(l):
    if netto_stromlast[q] > 0:
        y = 1
    elif netto_stromlast[q] <= 0:
        y = 0
    array_Autarkie_15_min = np.append(array_Autarkie_15_min, y)

LOLP = np.sum(array_Autarkie_15_min) / l
Autarkie_Jahr = 1 - LOLP

print('Die Autarkie für das gesamte Jahr Beträgt' + str(Autarkie_Jahr))

LOLP_pro_Tag = np.array([])
for i in range(0, l, len_day):
    array_y_96 = np.array([])
    for y in range(len_day):
        a = array_Autarkie_15_min[i + y]
        array_y_96 = np.append(array_y_96, a)
        LOLP = np.sum(array_y_96) / len_day
    LOLP_pro_Tag = np.append(LOLP_pro_Tag, LOLP)
    Autarkie_pro_Tag = 1 - LOLP_pro_Tag

min_Autarkie = np.min(Autarkie_pro_Tag)
max_Autarkie = np.max(Autarkie_pro_Tag)
min_Autarkie = round(min_Autarkie, 2)
max_Autarkie = round(max_Autarkie, 2)

print('Die geringste tägliche Autarkie des Jahres beträgt' + str(min_Autarkie))
print('Die höchste tägliche Autarkie des Jahres beträgt' + str(max_Autarkie))

LOLP_pro_Monat = np.array([])
for i in range(0, l, len_month):
    array_y_2920 = np.array([])
    for y in range(len_month):
        a = array_Autarkie_15_min[i + y]
        array_y_2920 = np.append(array_y_2920, a)
        LOLP = np.sum(array_y_2920) / len_month
    LOLP_pro_Monat = np.append(LOLP_pro_Monat, LOLP)
    Autarkie_pro_Monat = 1 - LOLP_pro_Monat
Autarkie_pro_Monat = np.around(Autarkie_pro_Monat, decimals=2)

plt.bar(np.arange(len(Autarkie_pro_Monat)), Autarkie_pro_Monat, width = 0.3, color = 'red')
#plt.title('Autarkie pro Monat')
plt.ylabel('Energieautonomie')
#plt.xlabel('Monate')
plt.xticks(np.arange(len(Autarkie_pro_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.axhline(1, 0, 12, color='orange')
plt.ylim((0, 1.1))
plt.savefig('Autarkie_Monat.png')
plt.show()


##pro Monat
list1 = []
for i in range(0, l, len_month):
    p_dem_2920 = np.array([])
    for j in range(len_month):
        cc = array_p_dem[i + j]
        p_dem_2920 = np.append(p_dem_96, cc)
    sum1 = np.sum(p_dem_2920)
    list1.append(sum1)
p_dem_ndarray_monat = np.asarray(list1)

list2 = []
for i in range(0, l, len_month):
    p_gen_2920 = np.array([])
    for j in range(len_month):
        cc = array_p_gen[i + j]
        p_gen_2920 = np.append(p_gen_2920, cc)
    sum2 = np.sum(p_gen_2920)
    list2.append(sum2)
p_gen_ndarray_monat = np.asarray(list2)

list3 = []
for i in range(0, l, len_month):
    BHKW_gen_2920 = np.array([])
    for j in range(len_month):
        cc = array_BHKW_gen[i + j]
        BHKW_gen_2920 = np.append(BHKW_gen_2920, cc)
    sum3 = np.sum(BHKW_gen_2920)
    list3.append(sum3)
BHKW_gen_ndarray_monat = np.asarray(list3)

list4 = []
for i in range(0, l, len_month):
    EV_2920 = np.array([])
    for j in range(len_month):
        cc = array_EV[i + j]
        EV_2920 = np.append(EV_2920, cc)
    sum4 = np.sum(EV_2920)
    list4.append(sum4)
EV_ndarray_monat = np.asarray(list4)

list5 = []
for i in range(0, l, len_month):
    Bat_einspeisen_2920 = np.array([])
    for j in range(len_month):
        cc = array_Bat_einspeisen[i + j]
        Bat_einspeisen_2920 = np.append(Bat_einspeisen_2920, cc)
    sum5 = np.sum(Bat_einspeisen_2920)
    list5.append(sum5)
Bat_einspeisen_ndarray_monat = np.asarray(list5)

list6 = []
for i in range(0, l, len_month):
    Bat_speichern_2920 = np.array([])
    for j in range(len_month):
        cc = array_Bat_speichern[i + j]
        Bat_speichern_2920 = np.append(Bat_speichern_2920, cc)
    sum6 = np.sum(Bat_speichern_2920)
    list6.append(sum6)
Bat_speichern_ndarray_monat = np.asarray(list6)

list7 = []
for i in range(0, l, len_month):
    HP_2920 = np.array([])
    for j in range(len_month):
        cc = array_HP[i + j]
        HP_2920 = np.append(HP_2920, cc)
    sum7 = np.sum(HP_2920)
    list7.append(sum7)
HP_ndarray_monat = np.asarray(list7)

list8 = []
for i in range(0, l, len_month):
    HR_2920 = np.array([])
    for j in range(len_month):
        cc = array_HR[i + j]
        HR_2920 = np.append(HR_2920, cc)
    sum8 = np.sum(HR_2920)
    list8.append(sum8)
HR_ndarray_monat = np.asarray(list8)




#Deckungsgrad
array_gamma_DG_monat = np.array([])
for i in range(number_of_months):

    if (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ) \
            > ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) ):
        gamma_DG = (( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) )
                    / (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] )) * 100
    elif ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]) ) \
            > (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ):
        gamma_DG = 100
    array_gamma_DG_monat = np.append(array_gamma_DG_monat, gamma_DG)
array_gamma_DG_monat = array_gamma_DG_monat.astype(int)


#Eigenverbrauch
array_gamma_EV_monat = np.array([])
for i in range(number_of_months):

    if ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])) \
            > (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ):
        gamma_EV = ((p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] )
                    / (np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i]))) *100
    elif ( np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])) == 0:
        gamma_EV = 0
    elif (p_dem_ndarray_monat[i] + HP_ndarray_monat[i] + HR_ndarray_monat[i] + EV_ndarray_monat[i] + Bat_speichern_ndarray_monat[i] ) \
            > (  np.absolute(p_gen_ndarray_monat[i]) + BHKW_gen_ndarray_monat[i] + np.absolute(Bat_einspeisen_ndarray_monat[i])):
        gamma_EV = 100
    array_gamma_EV_monat = np.append(array_gamma_EV_monat, gamma_EV)
array_gamma_EV_monat = array_gamma_EV_monat.astype(int)

###für gesamte Jahr

##Deckungsgrad
if (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)) \
        > (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))):
    gamma_DG_Jahr = ((np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen)))
                     / (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern))) * 100
elif (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) > \
            (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)):
        gamma_DG_Jahr = 100


##Eigenverbrauch
if (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) \
        > (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)):
    gamma_EV_Jahr = ((np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern))/(np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen)))) * 100
elif (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))) == 0:
    gamma_EV_Jahr = 0
elif (np.sum(array_p_dem) + np.sum(array_HP) + np.sum(array_HR) + np.sum(array_EV) + np.sum(array_Bat_speichern)) \
        > (np.absolute(np.sum(array_p_gen)) + np.sum(array_BHKW_gen) + np.absolute(np.sum(array_Bat_einspeisen))):
    gamma_EV_Jahr = 100

print('Der bilanzielle Deckungsgrad für das gesamte Jahr beträgt' + str(gamma_DG_Jahr))
print('Der bilanzielle Eigenverbrauch für das gesamte Jahr beträgt' + str(gamma_EV_Jahr))

##Deckungsgrad
gamma_DG_Jahr_real = np.array([])
Zähler = np.array([])
Nenner = np.array([])
for i in range(l):
    if (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) \
            > (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Zähler = np.append(Zähler, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
        Nenner = np.append(Nenner, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))

    elif (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) < (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Zähler = np.append(Zähler, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
        Nenner = np.append(Nenner, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
DG_Jahr = np.sum(Zähler)/np.sum(Nenner)
gamma_DG_monat_real = np.append(gamma_DG_Jahr_real, DG_Jahr*100)

##Eigenverbrauch
gamma_EV_Jahr_real = np.array([])
Zähler = np.array([])
Nenner = np.array([])
for i in range(l):
        if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) \
              > (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]):
            Zähler = np.append(Zähler, (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))

        elif (array_p_dem[i] + array_HP[i] + array_HR[i] + array_EV[i] + array_Bat_speichern[i]) > (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
EV_Jahr = np.sum(Zähler)/np.sum(Nenner)
gamma_EV_Jahr_real = np.append(gamma_EV_Jahr_real, EV_Jahr*100)



###Berechnung der realen Werte pro Monat
##Deckungsgrad
array_gamma_DG_monat_real = np.array([])
for i in range(0, l, len_month):
    Zähler = np.array([])
    Nenner = np.array([])
    for j in range(len_month):
        if (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) \
              > (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
            Nenner = np.append(Nenner, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))

        elif (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) < (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
            Nenner = np.append(Nenner, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
    DG_pro_Monat = np.sum(Zähler)/np.sum(Nenner)
    array_gamma_DG_monat_real = np.append(array_gamma_DG_monat_real, DG_pro_Monat*100)

##Eigenverbrauch
array_gamma_EV_monat_real = np.array([])
for i in range(0, l, len_month):
    Zähler = np.array([])
    Nenner = np.array([])
    for j in range(len_month):
        if (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])) \
              > (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]):
            Zähler = np.append(Zähler, (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))

        elif (array_p_dem[i + j] + array_HP[i + j] + array_HR[i + j] + array_EV[i + j] + array_Bat_speichern[i + j]) > (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])):
            Zähler = np.append(Zähler, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
            Nenner = np.append(Nenner, (np.absolute(array_p_gen[i + j]) + array_BHKW_gen[i + j] + np.absolute(array_Bat_einspeisen[i + j])))
    EV_pro_Monat = np.sum(Zähler)/np.sum(Nenner)
    array_gamma_EV_monat_real = np.append(array_gamma_EV_monat_real, EV_pro_Monat*100)


plt.bar(np.arange(len(array_gamma_DG_monat_real)),array_gamma_DG_monat_real, width=0.3, color = 'gray')
#plt.bar(np.arange(len(array_gamma_DG_monat))+0.15,array_gamma_DG_monat, width=0.3, color = 'gray')
#plt.plot(np.arange(len(DG_Monat_mean)), DG_min_monat, '.', color='black', markersize=12)
#plt.plot(np.arange(len(DG_Monat_mean)), DG_max_monat, '.', color='black', markersize=12)
#plt.title('Durchschnittlicher DG (blau) und \n bilanzieller DG(rot) pro Monat im Vergleich')
plt.ylabel('Deckungsgrad [%]')
#plt.xlabel('Monat')
plt.xticks(np.arange(len(array_gamma_DG_monat_real)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.axhline(100, 0, 12, color='orange')
plt.ylim((0, 110))
plt.savefig('Vergleich_Durchschnittlicher_bilanzieller_DG_Monat.png')
plt.show()

plt.bar(np.arange(len(array_gamma_EV_monat_real)),array_gamma_EV_monat_real, width=0.3, color = 'red')
#plt.bar(np.arange(len(array_gamma_EV_monat))+0.15,array_gamma_EV_monat, width=0.3, color = 'gray')
#plt.plot(np.arange(len(EV_Monat_mean)), EV_min_monat, '.', color='black', markersize=12)
#plt.plot(np.arange(len(EV_Monat_mean)), EV_max_monat, '.', color='black', markersize=12)
#plt.title('Durchschnittlicher EV (grau) und \n bilanzieller EV(dunkelrot) pro Monat im Vergleich')
plt.ylabel('Eigenverbauch [%]')
plt.axhline(100, 0, 12, color='orange')
#plt.xlabel('Monat')
plt.xticks(np.arange(len(array_gamma_EV_monat_real)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 110))
plt.savefig('Vergleich_Durchschnittlicher_bilanzieller_EV_Monat.png')
plt.show()



###   Last am Ortsnetztransformator   ###

Residuallast = ((array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern )\
			   - (np.absolute(array_p_gen) + array_BHKW_gen + np.absolute(array_Bat_einspeisen)))  # --> kW


Residuallast_w = np.array([])                  #Werte für P_GL für jeden Tag in einem array, dann 365 arrays in ein weiteres array
for i in range(0, l-len_day, len_week):
    Residuallast_Woche = np.array([])
    for j in range(len_week):
        cc = Residuallast[i + j]
        Residuallast_Woche = np.append(Residuallast_Woche, cc)
    Residuallast_w = np.append(Residuallast_w, Residuallast_Woche)
Residuallast_w  = np.reshape(Residuallast_w , (52, 168))

Residuallast_w_min = np.amin(Residuallast_w, axis=1)
Residuallast_w_max = np.amax(Residuallast_w, axis=1)

plt.bar(np.arange(len(Residuallast_w_min)),Residuallast_w_min, width=0.3, color = 'red')
plt.bar(np.arange(len(Residuallast_w_max)),Residuallast_w_max, width=0.3, color = 'gray')
plt.axhline(630, 0, 52, color='orange')
plt.axhline(-630, 0, 52, color='orange')
#plt.title('Maximale und minimale Residuallast des \n Quartiers innerhalb einer Woche')
#plt.xlabel('Woche')
plt.ylim((-900, 900))
plt.yticks((-900, -600, -300, 0, 300, 600, 900))
plt.ylabel('Last [kW]')
plt.savefig('Last_am_Ortsnetztransformator.png')
plt.show()

d = np.sum(Residuallast)

sum_Residualenergie = int(d /1000) #von kWh in MWh
print('Menge der Energie, die das Quartier bezieht(+)/abgibt(-)' + str(sum_Residualenergie) + ' MWh')

gradient = np.gradient(Residuallast)
min = gradient.min()
max = gradient.max()
print('Minimale Steigrate der Residuallast' + str(min))
print('Maximale Steigrate der Residuallast' + str(max))

plt.hist(gradient, bins=900, color='red', range=[-300, 300])
plt.ylabel('Anzahl stündlicher Messwerte')
plt.xlabel('Gradient [kW/h]')
plt.ylim((0, 200))
plt.xlim((-300,300))
plt.xticks(( -300, -200, -100, 0, 100, 200, 300))
#plt.title('Häufigkeitsverteilung der Gradienten')
plt.savefig('Häufigkeitsverteilung_der_Gradienten.png')
plt.show()

plt.hist(Residuallast, bins=800, color='red', range=[-800, 800])
#plt.title('Häufigkeitsverteilung der Residuallast im Quartier')
plt.ylabel('Anzahl stündlicher Messwerte')
plt.xlabel('Last [kW]')
plt.ylim((0,120))
plt.xlim((-800,600))
plt.xticks((-800, -600, -400, -200,  0, 200, 400, 600, 800))
#sns.distplot(Residuallast, norm_hist=False , kde=True, rug=True, bins=48, color = 'darkblue', rug_kws = {'linewidth': 3})
plt.savefig('Histogramm_Last_Ortsnetzstation.png')
plt.show()

###   Theoretischer Flexibilitätsbedarf des Quartiers   ###

Last_ohne_Speicher = ((array_p_dem + array_HP + array_HR + array_EV )\
			   - (np.absolute(array_p_gen) + array_BHKW_gen ))

list = []
for i in range(0, l, len_day):
    Last_ohne_Speicher_24 = np.array([])
    for j in range(len_day):
        a = Last_ohne_Speicher[i + j]
        Last_ohne_Speicher_24 = np.append(Last_ohne_Speicher_24, a)
    list.append(Last_ohne_Speicher_24)
Last_ohne_Speicher = np.asarray(list)     #in diesem Array wird die stündliche Last für jeden Tag festgehalten -> 365x24

Last_ohne_Speicher_mean = np.mean(Last_ohne_Speicher, axis=1)  #für jeden Tag wird die durchschnittliche Last als optimaler Lastverlauf bestimmt

plt.plot(Last_ohne_Speicher[0, :])
plt.axhline(Last_ohne_Speicher_mean[0], 0, 24, color='orange')
plt.show()

Last_ohne_Speicher_cumsum = np.cumsum(Last_ohne_Speicher, axis=1)  #BA Alicia: kumulierte Summe der Last für jeden Tag

Last_ohne_Speicher_mean_cumsum = np.array([])  #BA Alicia: kumulierte Summe der durchschnittl. Last für jeden Tag
for i in range(number_of_days):
    Last_ohne_Speicher_mean_cumsum = np.append(Last_ohne_Speicher_mean_cumsum, Last_ohne_Speicher_mean[i])
    for y in range(2, len_day+1):
        Last_ohne_Speicher_mean_cumsum = np.append(Last_ohne_Speicher_mean_cumsum, Last_ohne_Speicher_mean[i]*y)
Last_ohne_Speicher_mean_cumsum = np.reshape(Last_ohne_Speicher_mean_cumsum, (365, 24))

C_Flex = Last_ohne_Speicher_mean_cumsum - Last_ohne_Speicher_cumsum
C_Flex_min = np.absolute(np.min(C_Flex, axis=1))   #für jeden Tag wird die minimale Speicherkapazität bestimmt

C_Sto_theor = np.array([])
for i in range(number_of_days):
    a = C_Flex[i, :] + C_Flex_min[i]
    C_Sto_theor = np.append(C_Sto_theor, a)
C_Sto_theor = np.reshape(C_Sto_theor, (365, 24)) #stündlichen theor. Speicherladezustände für jeden Tag


C_Flex_theor = np.max(C_Sto_theor)   #theor Flexibilitätsbedarf eines "netzoptimalen" Betriebs


### tatsächlich vorhandene Speicherkapazitäten
number_batteries = 0
array_Umschaltpunkt_ges = np.zeros(365)
for q in range(120):

    loadFeed1 = np.zeros(l_35040)
    loadFeed2 = np.zeros(l_35040)

    for i in range(l_35040):
        if matrix_Bat[i, q] > 0:
            loadFeed1[i] = matrix_Bat[i, q]
        else:
            loadFeed2[i] = matrix_Bat[i, q]

    soc = np.zeros(l_35040)

    for t in range(l_35040):
        if t == 0:
            if np.sum(matrix_Bat[:, q]) == 0:
                SOC_previous = 0
            else: SOC_previous = 5
            number_batteries = number_batteries + 1
        else:
            SOC_previous = soc[t-1]

        soc[t] = ((SOC_previous + (0.25*(loadFeed1[t]*0.91 + loadFeed2[t]))) - (1e-4)*SOC_previous*0.25)

    gradient = np.gradient(matrix_Bat[:, q], axis=0)

    array_Umschaltpunkt = np.array([])
    for i in range(0, l_35040, 96):
        max_Umschaltpunkt = np.array([])
        array_Speicherstand_Bat2_Tag = np.array([])
        for j in range(95):
            array_Speicherstand_Bat2_Tag = np.append(array_Speicherstand_Bat2_Tag, soc[i + j])
            if gradient[i + j] > 0 and gradient[i + j +1] < 0:
                 max_Umschaltpunkt = np.append(max_Umschaltpunkt, soc[i + j])
            else: max_Umschaltpunkt = np.append(max_Umschaltpunkt, 0)
        if np.sum(max_Umschaltpunkt) == 0:
            array_Umschaltpunkt = np.append(array_Umschaltpunkt, np.max(array_Speicherstand_Bat2_Tag))
        else: array_Umschaltpunkt = np.append(array_Umschaltpunkt, np.max(max_Umschaltpunkt))

    array_Umschaltpunkt_ges = array_Umschaltpunkt_ges + array_Umschaltpunkt


array_Umschaltpunkt_ges = - np.sort(-array_Umschaltpunkt_ges)

plt.plot(array_Umschaltpunkt_ges, color='red')
plt.fill_between(np.arange(365), 0, array_Umschaltpunkt_ges, color='red', alpha= 0.5)
plt.ylabel('Speicherkapazität [kWh]')
plt.xlabel('Tage')
plt.xlim((0, 370))
plt.ylim((0, 1200))
plt.axhline(0.1*1200, 0, 365, color='orange')
plt.axhline(0.95*1200, 0, 365, color='orange')
plt.savefig('Speicherausnutzung_Bat.png')
plt.show()

area = np.trapz(array_Umschaltpunkt_ges, dx=1)
area = area - 0.1 * 1200 * 365
area_opti = 1200 * 365

Nutzungsgrad = (area / area_opti) *100

plt.plot(soc)
plt.show()

###   CO2-äquivalente Emissionen   ###

eta_netz = 0.9         #Verluste im Netz

total_rows = erzeugung.shape[0]   #Anzahl der Zeilen
total_columns = ABC.shape[1]

P_EE = np.absolute(array_p_gen) + array_BHKW_gen    #Erzeugung aus EE
Last_ges = array_p_dem + array_HP + array_HR + array_EV + array_Bat_speichern + array_Bat_einspeisen

Emissionsfaktor_PV = 0.101     #kg/kWh
Emissionsfaktor_BHKW = 0.4203    #Gas-BHKW  #https://www.umweltbundesamt.de/sites/default/files/medien/publikation/long/3476.pdf

array_Emissionsfaktor_EE = np.array([])
for i in range(l):
    if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]) == 0:
        Emissionsfaktor_EE = 0
    elif (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]) > 0:
        Emissionsfaktor_EE = Emissionsfaktor_PV * (np.absolute(array_p_gen[i]) / (np.absolute(array_p_gen[i]) + array_BHKW_gen[i])) + \
                             Emissionsfaktor_BHKW * (array_BHKW_gen[i] / (np.absolute(array_p_gen[i]) + array_BHKW_gen[i]))
    array_Emissionsfaktor_EE = np.append(array_Emissionsfaktor_EE, Emissionsfaktor_EE)

##Stromerzeugung DEA
array_Stromerzeugung_DEA_Monat = np.array([])
for i in range(0, l, len_month):
    Stromerzeugung_DEA_Monat = np.array([])
    for j in range(len_month):
        a = array_BHKW_gen[i + j] + np.absolute(array_p_gen[i + j])
        Stromerzeugung_DEA_Monat = np.append(Stromerzeugung_DEA_Monat, a)
    sum = np.sum(Stromerzeugung_DEA_Monat)
    array_Stromerzeugung_DEA_Monat = np.append(array_Stromerzeugung_DEA_Monat, sum)
array_Stromerzeugung_DEA_Monat = np.round(array_Stromerzeugung_DEA_Monat, decimals=0)

##Stromverbrauch pro Monat
array_Stromverbrauch_Monat = np.array([])
for i in range(0, l, len_month):
    Stromverbrauch_Monat = np.array([])
    for j in range(len_month):
        a = Last_ges[i + j]
        Stromverbrauch_Monat = np.append(Stromverbrauch_Monat, a)
    sum = np.sum(Stromverbrauch_Monat)
    array_Stromverbrauch_Monat = np.append(array_Stromverbrauch_Monat, sum)

array_Stromverbrauch_HP_Monat = np.array([])
for i in range(0, l, len_month):
    Stromverbrauch_HP_Monat = np.array([])
    for j in range(len_month):
        a = array_HP[i + j] + array_HR[i + j]
        Stromverbrauch_HP_Monat = np.append(Stromverbrauch_HP_Monat, a)
    sum = np.sum(Stromverbrauch_HP_Monat)
    array_Stromverbrauch_HP_Monat = np.append(array_Stromverbrauch_HP_Monat, sum)

##Energieverbrauch pro Monat
array_Energieverbrauch_Monat = np.array([])
for i in range(0, l, len_month):
    Energieverbrauch_Monat = np.array([])
    for j in range(len_month):
        a = Last_ges[i + j] + array_QDHW[i + j] + array_SH[i + j]
        Energieverbrauch_Monat = np.append(Energieverbrauch_Monat, a)
    sum = np.sum(Energieverbrauch_Monat)
    array_Energieverbrauch_Monat = np.append(array_Energieverbrauch_Monat, sum)
array_Energieverbrauch_Monat = np.round(array_Energieverbrauch_Monat, decimals=0)

##Energieverbrauch pro Tag
array_Energieverbrauch_day = np.array([])
for i in range(0, l, len_day):
    Energieverbrauch_day = np.array([])
    for j in range(len_day):
        a = Last_ges[i + j] + array_QDHW[i + j] + array_SH[i + j]
        Energieverbrauch_day = np.append(Energieverbrauch_day, a)
    sum = np.sum(Energieverbrauch_day)
    array_Energieverbrauch_day = np.append(array_Energieverbrauch_day, sum)

##Diagramm für Stromverbrauch pro Monat
plt.bar(np.arange(len(array_Stromverbrauch_Monat))-0.15, array_Stromverbrauch_Monat/1000, width=0.3, color = 'red')
plt.bar(np.arange(len(array_Stromverbrauch_Monat))+0.15, array_Stromerzeugung_DEA_Monat/1000, width=0.3, color = 'gray')
plt.ylabel('Elektrische Energie [MWh]')
#plt.xlabel('Monate')
plt.xticks(np.arange(len(array_Stromverbrauch_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 250))
plt.savefig('Stromverbrauch_pro_Monat.png')
plt.show()

##dynamischer CO2-Faktor des Quartiers
array_f_CO2_mix = np.array([])
for i in range(l):
    if (np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) >\
            (array_p_dem[i] + array_HR[i] + array_HP[i] + array_EV[i] + array_Bat_speichern[i]):
        f_CO2_mix = 0
        array_f_CO2_mix = np.append(array_f_CO2_mix, f_CO2_mix)
    else:
        array_P_f_CO2 = np.array([])
        array_P_ges = np.array([])
        for y in range(total_columns):
            P_ges = erzeugung[i,y]      #Erzeugte Leistung pro viertel Stunde  #Leistung in Erzeugungsdaten von Tennet in MWh
            P_f_CO2 = P_ges * ABC.iat[0,y] #el Leistung * CO2-äquivalente Emissionen  #2016 durchsch. CO2-Emissionsfaktor 0.516 kg/kWh
            array_P_f_CO2 = np.append(array_P_f_CO2, P_f_CO2)
            array_P_ges = np.append(array_P_ges, P_ges)
        b = np.sum(array_P_f_CO2)      #summe aller P_f_CO2 für einen Zeitunkt
        c = np.sum(array_P_ges)
        d = b / eta_netz
        f_CO2_mix = d / c      #[kg/kWh]
        array_f_CO2_mix = np.append(array_f_CO2_mix, f_CO2_mix)

array_f_CO2_mix_ohneEE = np.array([])    #dynamischer CO2-Faktor der Regelzone
for i in range(l):
    array_P_f_CO2_ohneEE = np.array([])
    array_P_ges_ohneEE = np.array([])
    for y in range(total_columns):
        P_ges = erzeugung[i,y]
        P_f_CO2 = P_ges * ABC.iat[0,y]
        array_P_f_CO2_ohneEE = np.append(array_P_f_CO2_ohneEE, P_f_CO2)
        array_P_ges_ohneEE = np.append(array_P_ges_ohneEE, P_ges)
    b2 = np.sum(array_P_f_CO2_ohneEE)
    c2 = np.sum(array_P_ges_ohneEE)
    d2 = b2 / eta_netz
    f_CO2_mix = d2 / c2
    array_f_CO2_mix_ohneEE = np.append(array_f_CO2_mix_ohneEE, f_CO2_mix)
Mittelwert_f_CO2_mix = np.mean(array_f_CO2_mix_ohneEE)          #statischer CO2-Faktor der Regelzone fürs gesamte Jahr


Last_Ortsnetztransformator = np.array([])    #Strom der ins Quartier bzw. aus dem Qiurtier fließt    #ohne Stromverbrauch der Wärmepumpen
Last_Ortsnetztransformator = (array_p_dem + array_EV + array_Bat_speichern + array_HP + array_HR)\
                    - (  np.absolute(array_p_gen) + array_BHKW_gen + np.absolute(array_Bat_einspeisen))

#array mit den Gesamtlasten in 2 arrays; Verbrauch und Einspeisung ins übergeordnete Netz
Last_einspeisen = np.array([])  #einspeisen ins übergeordnete Netz
Last_Strommix = np.array([])    #benötigter Strom wird mit Strommix gedeckt
for i in range(l):
    if Last_Ortsnetztransformator[i] > 0:
        Last_Strommix = np.append(Last_Strommix, Last_Ortsnetztransformator[i])
        Last_einspeisen = np.append(Last_einspeisen, 0)
    elif Last_Ortsnetztransformator[i] <= 0:
        Last_Strommix = np.append(Last_Strommix, 0)
        Last_einspeisen = np.append(Last_einspeisen, Last_Ortsnetztransformator[i])

array_Emissionen_HP = np.array([])
for i in range(l):
    if (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]+ array_HP[i] + array_HR[i]) < ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP = array_Emissionsfaktor_EE[i] * (array_HP[i] + array_HR[i])
    elif (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i] ) > ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP = array_f_CO2_mix[i] * (array_HP[i] + array_HR[i])
    elif (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]) < ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])):
        Emissionen_HP_EE = (( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])) - (array_p_dem[i] + array_EV[i] + array_Bat_speichern[i])) * array_Emissionsfaktor_EE[i]
        Emissionen_HP_Strommix = array_f_CO2_mix[i] * ((array_p_dem[i] + array_EV[i] + array_Bat_speichern[i]+ array_HP[i] + array_HR[i]) - ( np.absolute(array_p_gen[i]) + array_BHKW_gen[i] + np.absolute(array_Bat_einspeisen[i])))
        Emissionen_HP = Emissionen_HP_EE + Emissionen_HP_Strommix
    array_Emissionen_HP = np.append(array_Emissionen_HP, Emissionen_HP)

array_Emissionen_HP_Monat = np.array([])
for i in range(0, l, len_month):
    Emissionen_HP_Monat = np.array([])
    for j in range(len_month):
        a = array_Emissionen_HP[i + j]
        Emissionen_HP_Monat = np.append(Emissionen_HP_Monat, a)
    sum = np.sum(Emissionen_HP_Monat)
    array_Emissionen_HP_Monat = np.append(array_Emissionen_HP_Monat, sum)


array_Emissionen_EE_Verbrauch = np.array([])
for i in range(l):
    if P_EE[i] >= Last_ges[i] and Last_ges[i]>0:
        array_Emissionen_EE_Verbrauch = np.append(array_Emissionen_EE_Verbrauch, Last_ges[i])
    elif Last_ges[i] > P_EE[i]:
        array_Emissionen_EE_Verbrauch = np.append(array_Emissionen_EE_Verbrauch, P_EE[i])
    else: array_Emissionen_EE_Verbrauch = np.append(array_Emissionen_EE_Verbrauch, 0)


Emissionen_strommix = array_f_CO2_mix_ohneEE * Last_Strommix
list = []              #Emissionen monatlich verursacht durch den Strombezug aus Regelzone
for i in range(0, l, len_month):
    Emissionen_Monat = np.array([])
    for j in range(len_month):
        a = Emissionen_strommix[i + j]
        Emissionen_Monat = np.append(Emissionen_Monat, a)
    sum = np.sum(Emissionen_Monat)
    list.append(sum)
Emissionen_Strommix_m = np.asarray(list)

list = []              #Emissionen täglich verursacht durch den Strombezug aus Regelzone
for i in range(0, l, len_day):
    Emissionen_day = np.array([])
    for j in range(len_day):
        a = Emissionen_strommix[i + j]
        Emissionen_day = np.append(Emissionen_day, a)
    sum = np.sum(Emissionen_day)
    list.append(sum)
Emissionen_Strommix_day = np.asarray(list)

Emissionen_EE = array_Emissionen_EE_Verbrauch * array_Emissionsfaktor_EE
list = []              #Emissionen monatlich verursacht durch stromerzeugenden EE-Anlagen
for i in range(0, l, len_month):
    Emissionen_Monat_EE = np.array([])
    for j in range(len_month):
        a = Emissionen_EE[i + j]
        Emissionen_Monat_EE = np.append(Emissionen_Monat_EE, a)
    sum = np.sum(Emissionen_Monat_EE)
    list.append(sum)
Emissionen_Monatswerte_EE = np.asarray(list)

list = []              #Emissionen täglich verursacht durch stromerzeugenden EE-Anlagen
for i in range(0, l, len_day):
    Emissionen_day_EE = np.array([])
    for j in range(len_day):
        a = Emissionen_EE[i + j]
        Emissionen_day_EE = np.append(Emissionen_day_EE, a)
    sum = np.sum(Emissionen_day_EE)
    list.append(sum)
Emissionen_day_EE = np.asarray(list)


Emissionsfaktor_stat_month = np.array([])
Last_Strommix_month = np.array([])
for i in range(0, l, len_month):
    Emissionsfaktor_stat = np.array([])
    Last_Strommix_m = np.array([])
    for j in range(len_month):
        a = array_f_CO2_mix_ohneEE[i + j]
        Emissionsfaktor_stat = np.append(Emissionsfaktor_stat, a)
        Last_Strommix_m = np.append(Last_Strommix_m, Last_Strommix[i + j])
    mean_Emissionsfaktor_stat = np.mean(Emissionsfaktor_stat)
    Emissionsfaktor_stat_month = np.append(Emissionsfaktor_stat_month, mean_Emissionsfaktor_stat)
    Last_Strommix_month = np.append(Last_Strommix_month, np.sum(Last_Strommix_m))


Emissionen_Monatswerte_statisch_strommix = Last_Strommix_month * Emissionsfaktor_stat_month


#Deckung des monatlichen Strombedarfs mit statischem Emissionsfaktor
Emissionen_Monatswerte_statisch = Emissionen_Monatswerte_statisch_strommix + Emissionen_Monatswerte_EE #auf die monatlichen Emissionen müssen noch die für Deckungs des Wärmebedarfs, wenn keine HP oder CHP vorhanden


##monatlich verursachte Emissionen durch Deckung des Wärmebedarfs
array_Wärmebedarf_Gaskessel_ges = np.array([0,0,0,0,0,0,0,0,0,0,0,0])     #Keine BHKW oder Wärmepumpen im Quartier -> gesamter Wärmebedarf wird durch Gaskessel gedeckt
for q in range(120):
    array_Wärmebedarf_monat = np.array([])
    if np.sum(matrix_BHKW_gen2, axis=0)[q] + np.sum(matrix_HP2, axis=0)[q] == 0:
        for i in range(0, l, len_month):
            Wärmebedarf_monat = np.array([])
            for j in range(len_month):
                a = matrix_SH2[i+j, q] + matrix_QDHW2[i+j, q]
                Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
            sum = np.sum(Wärmebedarf_monat)
            array_Wärmebedarf_monat = np.append(array_Wärmebedarf_monat, sum)
        array_Wärmebedarf_Gaskessel_ges = array_Wärmebedarf_Gaskessel_ges + array_Wärmebedarf_monat
Emissionen_Wärme_Gaskessel = array_Wärmebedarf_Gaskessel_ges * 0.25/0.77

array_Wärmebedarf_BHKW_ges = np.array([0,0,0,0,0,0,0,0,0,0,0,0])   #Wärmebedarf wird vollständig mit BHKW gedeckt
for q in range(120):
    array_Wärmebedarf_monat = np.array([])
    if np.sum(matrix_BHKW_gen2, axis=0)[q] > 0:
        for i in range(0, l, len_month):
            Wärmebedarf_monat = np.array([])
            for j in range(len_month):
                a = matrix_SH2[i+j, q] + matrix_QDHW2[i+j, q]
                Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
            sum = np.sum(Wärmebedarf_monat)
            array_Wärmebedarf_monat = np.append(array_Wärmebedarf_monat, sum)
        array_Wärmebedarf_BHKW_ges = array_Wärmebedarf_BHKW_ges + array_Wärmebedarf_monat
Emissionen_Wärme_BHKW = array_Wärmebedarf_BHKW_ges * 0.1956    #0.1956 kg/kWh CO2e durch Gas-BHKW für Wärmeproduktion

Emissionen_Gas_real = Emissionen_Wärme_Gaskessel + Emissionen_Wärme_BHKW  #Verursachte Emissionen durch Gaskessel und BHKW

##täglich verursachte Emissionen durch Deckung des Wärmebedarfs
array_Wärmebedarf_Gaskessel_day = np.zeros(number_of_days, dtype=int)     #Keine BHKW oder Wärmepumpen im Quartier -> gesamter Wärmebedarf wird durch Gaskessel gedeckt
for q in range(120):
    array_Wärmebedarf_day = np.array([])
    if np.sum(matrix_BHKW_gen2, axis=0)[q] + np.sum(matrix_HP2, axis=0)[q] == 0:
        for i in range(0, l, len_day):
            Wärmebedarf_day = np.array([])
            for j in range(len_day):
                a = matrix_SH2[i+j, q] + matrix_QDHW2[i+j, q]
                Wärmebedarf_day = np.append(Wärmebedarf_day, a)
            sum = np.sum(Wärmebedarf_day)
            array_Wärmebedarf_day = np.append(array_Wärmebedarf_day, sum)
        array_Wärmebedarf_Gaskessel_day = array_Wärmebedarf_Gaskessel_day + array_Wärmebedarf_day
Emissionen_Wärme_Gaskessel_day = array_Wärmebedarf_Gaskessel_day * 0.25/0.77

array_Wärmebedarf_BHKW_day = np.zeros(number_of_days, dtype=int)    #Wärmebedarf wird vollständig mit BHKW gedeckt
for q in range(120):
    array_Wärmebedarf_day = np.array([])
    if np.sum(matrix_BHKW_gen2, axis=0)[q] > 0:
        for i in range(0, l, len_day):
            Wärmebedarf_day = np.array([])
            for j in range(len_day):
                a = matrix_SH2[i+j, q] + matrix_QDHW2[i+j, q]
                Wärmebedarf_day = np.append(Wärmebedarf_day, a)
            sum = np.sum(Wärmebedarf_day)
            array_Wärmebedarf_day = np.append(array_Wärmebedarf_day, sum)
        array_Wärmebedarf_BHKW_day = array_Wärmebedarf_BHKW_day + array_Wärmebedarf_day
Emissionen_Wärme_BHKW_day = array_Wärmebedarf_BHKW_day * 0.1956    #0.1956 g/kWh CO2e durch Gas-BHKW für Wärmeproduktion

Emissionen_Gas_real_day = Emissionen_Wärme_Gaskessel_day + Emissionen_Wärme_BHKW_day  #Verursachte Emissionen durch Gaskessel und BHKW


##
array_Wärmebedarf_Gaskessel_theo = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
for q in range(120):
    array_Wärmebedarf_monat = np.array([])
    if np.sum(matrix_HP2, axis=0)[q] == 0:
        array_Wärmebedarf_monat = np.array([])
        for i in range(0, l, len_month):
            Wärmebedarf_monat = np.array([])
            for j in range(len_month):
                a = matrix_SH2[i+j, q] + matrix_QDHW2[i+j, q]
                Wärmebedarf_monat = np.append(Wärmebedarf_monat, a)
            sum = np.sum(Wärmebedarf_monat)
            array_Wärmebedarf_monat = np.append(array_Wärmebedarf_monat, sum)
        array_Wärmebedarf_Gaskessel_theo = array_Wärmebedarf_Gaskessel_theo + array_Wärmebedarf_monat
Emissionen_Gas_theo = array_Wärmebedarf_Gaskessel_theo * 0.25/0.77


Emissionen_vergleich_ges = Emissionen_Monatswerte_statisch + Emissionen_Gas_theo
#Wirkungsgrad Gaskessel 77%  #https://www.haustechnikdialog.de/SHKwissen/1888/Wirkungs-und-Nutzungsgrad-einer-Heizungsanlage
#CO2e von Erdgas 250 g/kWh

##Min und Max Emissionsfaktor des Quartiers
Emissionsfaktor_Quartier_max = np.max((Emissionen_Gas_real_day + Emissionen_day_EE + Emissionen_Strommix_day)/ array_Energieverbrauch_day)
Emissionsfaktor_Quartier_min = np.min((Emissionen_Gas_real_day + Emissionen_day_EE + Emissionen_Strommix_day)/ array_Energieverbrauch_day)
Emissionsfaktor_Quartier_year = ((np.sum(Emissionen_Gas_real_day) + np.sum(Emissionen_day_EE) + np.sum(Emissionen_Strommix_day)) / np.sum(array_Energieverbrauch_day))

Emissionen_ges = np.sum(Emissionen_strommix) + np.sum(Emissionen_EE) + np.sum(Emissionen_Gas_real)  #kg    #Emissionen die sich aus dem CO2-Faktor des Quartiers und der Gesamtlast/-nachfrage ergeben
Emissionen_ges = round(Emissionen_ges)
print('Verursachte Emissionen im Jahr in kg' + str(Emissionen_ges))

##Einsparungen durch Einspeisung
Fall_1 = array_f_CO2_mix_ohneEE * np.absolute(Last_einspeisen)     #Einsparungen, die sich aus der Einspeisung des Quartiers in die Regelzone
Fall_1_stat = Mittelwert_f_CO2_mix * np.absolute(Last_einspeisen)  # und der Differenz der CO2-Faktoren ergeben
Fall_2 = array_Emissionsfaktor_EE * np.absolute(Last_einspeisen)
list = []              #Einsparung pro Monat dynamisch
for i in range(0, l, len_month):
    Einsparung_pro_Monat = np.array([])
    for j in range(len_month):
        a =  Fall_1[i + j] - Fall_2[i + j]
        Einsparung_pro_Monat  = np.append(Einsparung_pro_Monat , a)
    sum = np.sum(Einsparung_pro_Monat)
    list.append(sum)
Einsparung_Monate = np.asarray(list)
list = []              #Einsparung pro Monat statisch
for i in range(0, l, len_month):
    Einsparung_pro_Monat = np.array([])
    for j in range(len_month):
        a =  Fall_1_stat[i + j] - Fall_2[i + j]
        Einsparung_pro_Monat  = np.append(Einsparung_pro_Monat , a)
    sum = np.sum(Einsparung_pro_Monat)
    list.append(sum)
Einsparung_Monate_stat = np.asarray(list)

Einsparung_Monate_stat_pos = np.array([])
Einsparung_Monate_stat_neg = np.array([])
Einsparung_Monate_pos = np.array([])
Einsparung_Monate_neg = np.array([])
for i in range(12):
    if Einsparung_Monate_stat[i] > 0:
        Einsparung_Monate_stat_pos = np.append(Einsparung_Monate_stat_pos, Einsparung_Monate_stat[i])
        Einsparung_Monate_stat_neg = np.append(Einsparung_Monate_stat_neg, 0)
    elif Einsparung_Monate_stat[i] <= 0:
        Einsparung_Monate_stat_pos = np.append(Einsparung_Monate_stat_pos, 0)
        Einsparung_Monate_stat_neg = np.append(Einsparung_Monate_stat_neg, Einsparung_Monate_stat[i])
for i in range(12):
    if Einsparung_Monate[i] > 0:
        Einsparung_Monate_pos = np.append(Einsparung_Monate_pos, Einsparung_Monate[i])
        Einsparung_Monate_neg = np.append(Einsparung_Monate_neg, 0)
    elif Einsparung_Monate[i] <= 0:
        Einsparung_Monate_pos = np.append(Einsparung_Monate_pos, 0)
        Einsparung_Monate_neg = np.append(Einsparung_Monate_neg, Einsparung_Monate[i])


##min und max Emissionen bei Strombezug des Quartiers zu günstigen/ungünstigen Zeitpunkten bzgl. CO2e-Faktor

array_Last_Strommix_d = np.array([])  #Deckung der Last an Ortsnetzstation durch Strom aus Regelzone
Strombezug_max = np.array([])
for i in range(0, l, len_month):
    Last_Strommix_d = np.array([])
    for j in range(len_month):
        c = Last_Strommix[i + j]
        Last_Strommix_d = np.append(Last_Strommix_d, c)
    max_d = np.max(Last_Strommix_d)
    Strombezug_max = np.append(Strombezug_max, max_d)
    sum_d = np.sum(Last_Strommix_d)
    array_Last_Strommix_d = np.append(array_Last_Strommix_d, sum_d)  # Strombezug des Quartiers aus Regelzone pro tag für 365 tage

h_need = array_Last_Strommix_d / Strombezug_max  #Anzahl der benötigten Stunden, um Strombezug bei maximaler Last zu decken

h_fl_round_up = np.ceil(h_need)   #aufrunden für max Emissionen
h_fl_round_up = h_fl_round_up.astype(int)
h_fl_round_down = np.floor(h_need)   #aufrunden für min Emissionen
h_fl_round_down = h_fl_round_down.astype(int)

h = -1
array_pos_min = np.array([])
array_pos_max = np.array([])
for j in range(0, l, len_month):
    h = h + 1
    x_up = h_fl_round_up[h]
    x_up = int(x_up)
    x_down = h_fl_round_down[h]
    x_down = int(x_down)

    array_CO2e_best_worst2 = np.array([])
    for i in range(len_month):
            z = array_f_CO2_mix_ohneEE[i + j]
            array_CO2e_best_worst2 = np.append(array_CO2e_best_worst2, z)
    array_CO2e_best_worst2_sorted = np.sort(array_CO2e_best_worst2)
    array_CO2e_best_worst2_sorted = np.unique(array_CO2e_best_worst2_sorted)

    pos_min_values = np.array([])
    for q in range(x_down):
        pos_min = np.where(array_CO2e_best_worst2 == array_CO2e_best_worst2_sorted[q])
        pos_min = np.asarray(pos_min)
        pos_min_values = np.append(pos_min_values, pos_min[0] + h * len_month)#[:h_fl_round_up[h]] + h * len_month)
        #if len(pos_min_values) >= h_fl_round_up[h]:
            #pos_min_values = pos_min_values[:h_fl_round_up[h]]
            #break
    array_pos_min = np.append(array_pos_min, pos_min_values)

    pos_max_values = np.array([])
    for q in range(x_up):
        pos_max = np.where(array_CO2e_best_worst2 == array_CO2e_best_worst2_sorted[-q - 1])
        pos_max = np.asarray(pos_max)
        pos_max_values = np.append(pos_max_values, pos_max[0] + h * len_month)#[:h_fl_round_up[h]] + h * len_month)
        #if len(pos_max_values) >= h_fl_round_up[h]:
            #pos_max_values = pos_max_values[:h_fl_round_up[h]]
            #break
    array_pos_max = np.append(array_pos_max, pos_max_values)

Last_verlegt_max = np.zeros(l, dtype=int)
Last_verlegt_min = np.zeros(l, dtype=int)

m = 0
for i in range(12):
    for j in range(h_fl_round_up[i]):
        array_max = np.put(Last_verlegt_max, (array_pos_max[m + j]), Strombezug_max[i])
    m = m + h_fl_round_up[i]

m = 0
for i in range(12):
    for j in range(h_fl_round_down[i]):
        array_min = np.put(Last_verlegt_min, (array_pos_min[m + j]), Strombezug_max[i])
    m = m + h_fl_round_down[i]

Emissionen_max = np.sum(Last_verlegt_max * array_f_CO2_mix_ohneEE) + np.sum(Emissionen_EE) + np.sum(Emissionen_Gas_real)
Emissionen_min = np.sum(Last_verlegt_min * array_f_CO2_mix_ohneEE) + np.sum(Emissionen_EE) + np.sum(Emissionen_Gas_real)


list = []              #min Emissionen pro Monat für günstigen Strombezug aus Regelzone
for i in range(0, l, len_month):
    Emissionen_min_Monat = np.array([])
    for j in range(len_month):
        a = Last_verlegt_min[i + j] * array_f_CO2_mix_ohneEE[i + j]
        Emissionen_min_Monat = np.append(Emissionen_min_Monat, a)
    sum = np.sum(Emissionen_min_Monat)
    list.append(sum)
Emissionen_min_array = np.asarray(list)

list = []              #max Emissionen pro Monat für ungünstigen Strombezug aus Regelzone
for i in range(0, l, len_month):
    Emissionen_max_Monat = np.array([])
    for j in range(len_month):
        a = Last_verlegt_max[i + j] * array_f_CO2_mix_ohneEE[i + j]
        Emissionen_max_Monat = np.append(Emissionen_max_Monat, a)
    sum = np.sum(Emissionen_max_Monat)
    list.append(sum)
Emissionen_max_array = np.asarray(list)

bar1 = Emissionen_Monatswerte_EE + Emissionen_Strommix_m + Emissionen_Gas_real
bar2 = array_Energieverbrauch_Monat
plt.bar(np.arange(len(array_Energieverbrauch_Monat))-0.15, bar1/1000, width=0.3, color = 'red')
plt.title('Energieverbrauch und Gesamtemissionen')
plt.ylabel('Emissionen [t]')
plt.yticks((0, 25, 50, 75, 100, 125, 150, 175, 200))
plt.ylim((0, 200))
plt.twinx()
plt.ylabel('Energieverbrauch [MWh]')
plt.yticks((0, 100, 200, 300, 400, 500, 600, 700))
plt.ylim((0, 700))
plt.bar(np.arange(len(array_Energieverbrauch_Monat))+0.15, bar2/1000, width=0.3, color = 'gray')   #MWh
#plt.xlabel('Monat')
plt.xticks(np.arange(len(array_Energieverbrauch_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.savefig('Energieverbrauch_Gesamtemissionen.png')
plt.show()

plt.bar(np.arange(len(array_Energieverbrauch_Monat)), (bar1 / bar2), width=0.3, color = 'orange')
plt.title('Emissionsfaktor des Quartiers')
plt.ylabel('Emissionsfaktor [kg/kWh]')
plt.axhline(0.101, 0, 12, color='k')   #Photovoltaik
plt.axhline(0.428, 0, 12, color='k')   #Erdgaskraftwerk
plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
plt.ylim((0, 1))
plt.xticks(np.arange(len(array_Energieverbrauch_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.savefig('Emissionsfaktor_Quartier.png')
plt.show()

bar3 = Einsparung_Monate_pos - Einsparung_Monate_neg
bar4 = Einsparung_Monate_stat_pos - Einsparung_Monate_stat_neg
plt.bar(np.arange(len(Einsparung_Monate_pos))-0.15, bar3/1000, width=0.3, color = 'orange')
plt.bar(np.arange(len(Einsparung_Monate_pos))+0.15, bar4/1000, width=0.3, color = 'tan')
plt.title('Einsparungen des Quartiers')
plt.ylabel('Eingesparte Emissionen [t]')
plt.yticks(( 0, 10, 20, 30, 40))
plt.ylim((0, 40))
plt.xticks(np.arange(len(array_Energieverbrauch_Monat)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.savefig('Einsparungen_Quartier.png')
plt.show()


points_1 = Emissionen_Monatswerte_EE + Emissionen_max_array
points_2 = Emissionen_Monatswerte_EE + Emissionen_min_array
bar_1 = Emissionen_Monatswerte_EE + Emissionen_Strommix_m
bar_4 = Emissionen_Monatswerte_statisch

plt.plot(np.arange(len(Emissionen_Strommix_m))-0.15, points_1/1000, color='k', marker='.', markersize=6, linewidth=0)
plt.plot(np.arange(len(Emissionen_Strommix_m))-0.15, points_2/1000, color='k', marker='.', markersize=6, linewidth=0)
plt.bar(np.arange(len(Emissionen_Strommix_m))-0.15, bar_1/1000, width=0.3, color = 'red')
plt.bar(np.arange(len(Emissionen_Strommix_m))+0.15, bar_4/1000, width=0.3, color = 'gray')
#plt.title('Emissionen mit dynamischen Emissionsfaktor')
plt.ylabel('Emissionen [t]')
plt.yticks(( 0, 25, 50, 75, 100, 125, 150))
#plt.xlabel('Monat')
plt.xticks(np.arange(len(Emissionen_Strommix_m)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 150))
plt.savefig('CO2_äquivalente_Emissionen_mit_dynamischen_Emissionsfaktor.png')
plt.show()


bar_5 = Emissionen_Gas_real
bar_6 = Emissionen_Gas_theo
plt.bar(np.arange(len(Emissionen_Gas_real))-0.15, bar_5/1000, width=0.3, color = 'red')
plt.bar(np.arange(len(Emissionen_Gas_real))+0.15, bar_6/1000, width=0.3, color = 'gray')
#plt.title('Emissionen mit dynamischen Emissionsfaktor')
plt.ylabel('Emissionen [t]')
plt.yticks(( 0, 25, 50, 75, 100, 125, 150))
#plt.xlabel('Monat')
plt.xticks(np.arange(len(Emissionen_Strommix_m)), ['Jan', '', 'März', '', 'Mai', '', 'Jul', '', 'Sep', '', 'Nov', ''])
plt.ylim((0, 150))
plt.savefig('CO2_Emissionen_Gas.png')
plt.show()

Emissionen_Mittelwert_strommix = Mittelwert_f_CO2_mix * Last_Strommix + array_Emissionen_EE_Verbrauch * array_Emissionsfaktor_EE
Emissionen_ges2 = np.sum(Emissionen_Mittelwert_strommix)  #kg
Emissionen_ges2 = round(Emissionen_ges2)    #Emissionen mit gemitteltem CO2 Faktor


###Einsparung durch zu viel EE-Erzeugung im Quartier

Einsparung = np.sum(Fall_1) - np.sum(Fall_2)
Einsparung = int(Einsparung)
print('Die Einsparungen durch zu viel EE-Erzeugung beträgt' + str(Einsparung) + ' kg')


###Bewertungsmatrix
## Emissionen
percent = np.zeros(number_of_months, dtype=int)
value_emission = np.array([])
for i in range(number_of_months):
    if bar_1[i] >= bar_4[i]:
        percent[i] = (bar_1[i] - bar_4[i]) / (points_1[i] - bar_4[i]) * 100
    else: percent[i] = (bar_1[i] - points_2[i]) / (bar_4[i] - points_2[i]) * 100 * -1
    if percent[i] >= 66:
        value_emission = np.append(value_emission, 3)
    elif percent[i] >= 33 and percent[i] < 66:
        value_emission = np.append(value_emission, 2)
    elif percent[i] >= 0 and percent[i] < 33:
        value_emission = np.append(value_emission, 1)
    elif percent[i] < 0 and percent[i] >= -33:
        value_emission = np.append(value_emission, -1)
    elif percent[i] < -33 and percent[i] >= -66:
        value_emission = np.append(value_emission, -2)
    elif percent[i] < -66 and percent[i] >= -100:
        value_emission = np.append(value_emission, -3)

##Ausnutzung Flexi

C_Sto_tatsächlich = number_batteries * 10   #kWh

if Nutzungsgrad >= 0 and Nutzungsgrad < 25:
    value_ausnutzung = 1
elif Nutzungsgrad >= 25 and Nutzungsgrad < 50:
    value_ausnutzung = 2
elif Nutzungsgrad >= 50:
    value_ausnutzung = 3
elif Nutzungsgrad < 0:
    value_ausnutzung = 0

if C_Sto_tatsächlich > C_Flex_theor:
    value_ausnutzung = value_ausnutzung - 4




###PDF Erzeugung
c = canvas.Canvas('Ergebnisse_Systemische_Bewertung.pdf')
c.setFont('Helvetica', 11)
c.line(0.5*cm, 27.5*cm, 20.5*cm, 27.5*cm)
c.drawImage('Logo_EON.png', 14*cm, 28*cm, 176, 30)
c.drawImage('Stromverbrauch_pro_Monat.png', 1*cm, 18*cm, 320, 240)
c.drawString(1*cm, 17*cm, 'Die monatlichen Stromverbräuche betragen ' + str(np.round(array_Stromverbrauch_Monat/1000, decimals=0)) + ' MWh')
c.drawString(1*cm, 16*cm,'Monatliche EE-Erzeugung: ' + str(np.round(array_Stromerzeugung_DEA_Monat/1000, decimals=0)) + ' MWh')
c.drawImage('Energieverbrauch_Gesamtemissionen.png', 1*cm, 7*cm, 320, 240)
c.drawString(1*cm, 6*cm, 'Der Energieverbrauch für das gesamte Jahr beträgt: ' + str(np.round(np.sum(array_Energieverbrauch_Monat/1000))) + ' MWh')
c.drawString(1*cm, 5*cm,'EE-Erzeugung im Jahr: ' + str(np.round(np.sum(array_Stromerzeugung_DEA_Monat/1000), decimals=0)) + ' MWh')
c.drawString(1*cm, 4*cm, 'Die monatlichen Energieverbräuche Jahr betragen: ' + str(np.round(array_Energieverbrauch_Monat/1000, decimals=0)) + ' MWh')
c.drawString(1*cm, 3*cm,'Monatliche EE-Erzeugung: ' + str(np.round(array_Stromerzeugung_DEA_Monat/1000, decimals=0)) + ' MWh')
c.showPage()
c.drawImage('Emissionsfaktor_Quartier.png', 1*cm, 18*cm, 320, 240)
c.drawString(1*cm, 17*cm, 'Die monatlichen Emissionsfaktoren betragen: ' + str(np.round((bar1 / bar2), decimals=0)))
c.drawImage('CO2_äquivalente_Emissionen_mit_dynamischen_Emissionsfaktor.png', 1*cm, 8*cm, 320, 240)
c.drawString(1*cm, 7*cm, 'Die Emissionen für das gesamte Jahr betragen ' + str(Emissionen_ges) + ' kg')
c.drawString(1*cm, 6*cm,'Die Einsparungen durch zu viel EE-Erzeugung beträgt ' + str(Einsparung) + ' kg')
c.showPage()

c.drawImage('Einsparungen_Quartier.png', 1*cm, 18*cm, 320, 240)
c.showPage()

c.drawImage('Speicherausnutzung_Bat.png', 1*cm, 18*cm, 320, 240)
c.drawString(1*cm, 17*cm, 'Der Ausnutzungsgrad der Batteriespeicher beträgt: ' + str(np.round(Nutzungsgrad, decimals=0) ))
c.drawString(1*cm, 16*cm, 'Theoretisch benötigte Flexibilität für netzopt. Betrieb: ' + str(np.round(C_Flex_theor, decimals=0) ))
c.showPage()

c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_Ortsnetzstation.png', 1*cm, 20*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Leistungsaufnahmefähigkeit_Ortsnetzstation.png', 1*cm, 12*cm, 320, 240)
c.drawString(1*cm, 8*cm, 'Die Engpassarbeit an der Ortnetzstation, die durch Erzeugung entsteht, beträgt ' + str(Engpassarbeit_neg) + ' kWh')
#c.drawString(1*cm, 7*cm, 'Engpassgefahr durch Last besteht, wenn Wert negativ ist: ' + str(Engpassgefahr_Last))
#c.drawString(1*cm, 6*cm, 'Engpassgefahr durch Erzeugung besteht, wenn Wert negativ ist: ' + str(Engpassgefahr_Erzeugung))
c.showPage()

c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 1.png', 1*cm, 20*cm, 240, 180)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 2.png', 1*cm, 12*cm, 240, 180)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 3.png', 1*cm, 4*cm, 240, 180)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 4.png', 11*cm, 20*cm, 240, 180)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 5.png', 11*cm, 12*cm, 240, 180)
c.drawImage('Häufigkeitsverteilung_der_Engpassleistung_von_Strang 6.png', 11*cm, 4*cm, 240, 180)
c.showPage()

c.drawImage('Vergleich_Durchschnittlicher_bilanzieller_DG_Monat.png', 1*cm, 19*cm, 240, 180)
c.drawImage('Vergleich_Durchschnittlicher_bilanzieller_EV_Monat.png', 111*cm, 19*cm, 240, 180)
c.drawString(1*cm, 29*cm, 'Werte bilanzielle DG: ' + str(array_gamma_DG_monat))
c.drawString(1*cm, 28*cm, 'Werte bilanzielle DG: ' + str(array_gamma_EV_monat))
c.drawString(1*cm, 11*cm, 'Werte durchschnittlicher EV: ' + str(array_gamma_EV_monat_real))
c.drawString(1*cm, 10*cm, 'Werte durchschnittlicher DG: ' + str(array_gamma_DG_monat_real))
c.drawString(1*cm, 9*cm,'Der bilanzielle Deckungsgrad für das gesamte Jahr beträgt ' + str(gamma_DG_Jahr))
c.drawString(1*cm, 8*cm,'Der reale DG für das Jahr beträgt' + str(gamma_DG_Jahr_real))
c.drawString(1*cm, 7*cm,'Der bilanzielle Eigenverbrauch für das gesamte Jahr beträgt ' + str(gamma_EV_Jahr))
c.drawString(1*cm, 6*cm,'Der reale EV für das Jahr beträgt' + str(gamma_EV_Jahr_real))
c.drawString(1*cm, 5*cm,'Der minimale tägliche Deckungsgrad der Monate beträgt' + str(DG_min_monat))
c.drawString(1*cm, 4*cm,'Der maximale tägliche Deckungsgrad der Monate beträgt' + str(DG_max_monat))
c.drawString(1*cm, 3*cm,'Der minimale tägliche Eigenverbrauch der Monate beträgt' + str(EV_min_monat))
c.drawString(1*cm, 2*cm,'Der maximale tägliche Eigenverbrauch der Monate beträgt' + str(EV_min_monat))
c.showPage()
c.drawImage('Autarkie_Monat.png', 1*cm, 20*cm, 320, 240)
c.drawString(1*cm, 18*cm, 'Die Autarkie für das gesamte Jahr Beträgt ' + str(Autarkie_Jahr))
c.drawString(1*cm, 17*cm,'Die geringste tägliche Autarkie des Jahres beträgt ' + str(min_Autarkie))
c.drawString(1*cm, 16*cm,'Die höchste tägliche Autarkie des Jahres beträgt ' + str(max_Autarkie))
c.drawString(1*cm, 10*cm, 'Monatliche Autarkien: ' + str(Autarkie_pro_Monat))
c.showPage()
#c.drawImage('GSC_abs.png', 1*cm, 20*cm, 320, 240)
#c.drawImage('GSC_rel.png', 1*cm, 12*cm, 320, 240)
c.showPage()
c.drawImage('Last_am_Ortsnetztransformator.png', 1*cm, 20*cm, 320, 240)
c.drawString(1*cm, 3*cm, 'Menge der Energie, die das Quartier bezieht(+)/abgibt(-)' + str(sum_Residualenergie) + ' MWh')
c.drawImage('Histogramm_Last_Ortsnetzstation.png', 1*cm, 12*cm, 320, 240)
c.drawImage('Häufigkeitsverteilung_der_Gradienten.png', 1*cm, 4*cm, 320, 240)
c.showPage()
#c.drawImage('Benötigte_Speicher_pro_Woche.png', 1*cm, 20*cm, 320, 240)
#c.drawImage('Benötigter_Speicher_pro_Monat.png', 1*cm, 12*cm, 320, 240)
#c.drawString(1*cm, 11*cm, 'Maximaler Speicherbedarf für Tageszyklus, wenn 90% abgedeckt werden sollen ' + str(AA) + ' MWh')
#c.drawString(1*cm, 10*cm, 'Maximaler Speicherbedarf für Tageszyklus ' + str(storage_needed_d_max) + ' MWh')
#c.drawString(1*cm, 9*cm, 'Maximaler Speicherbedarf für Wochenzyklus ' + str(storage_needed_w_max) + ' MWh')
#c.drawString(1*cm, 8*cm, 'Maximaler Speicherbedarf für Wochenzyklus, wenn 90% abgedeckt werden sollen ' + str(storage_needed_w_max_sorted)+ ' MWh')
#c.drawString(1*cm, 7*cm, 'Maximaler Speicherbedarf für Monatszyklus, wenn 90% abgedeckt werden sollen ' + str(storage_needed_m_max_sorted) + ' MWh')
#c.drawString(1*cm, 6*cm, 'Maximaler Speicherbedarf für Monatszyklus ' + str(storage_needed_m_max)+ ' MWh')
#c.drawString(1*cm, 5*cm, 'Die Speicherkapazität für den Betrachtungszeitraum von einem Jahr würde' + str(storage_needed_j) + ' MWh betragen')
c.save()

root.destroy()