import Persistence.Reader as rd
import DataAnalysis.DataAnalysisClass as da

reader = rd.CSVReader
analysis = da.DataAnalysisClass

directors, actors, data = reader.read("../cleaned_data.csv")
analysis.analyze(directors, actors, data)