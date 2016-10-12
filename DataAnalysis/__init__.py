import Persistence.Reader as rd
import DataAnalysis.DataAnalysisClass as da

reader = rd.CSVReader
analysis = da.DataAnalysisClass

analysis.analyze(reader.read("../cleaned_data.csv"))