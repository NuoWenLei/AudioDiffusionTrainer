import pandas as pd
import numpy as np
import datasets
from datasets import Dataset, Audio

class DataLoader():
	def __init__(self,
	      csvPath: str,
		  audioDirPath: str,
		  audioColumn: str,
		  newAudioColumn: str = "audioArray",
		  targetColumn: str = None,
		  samplingRate: int = 30000,
		  numSamples: int = 2**18,
	      batchSize: int = 32,
	      returnDataset: bool = False,
		  shuffle: bool = True,
		  bucketColumn: str = None):
		self.sampleCsv = pd.read_csv(csvPath)
		self.targetColumn = targetColumn
		self.samplingRate = samplingRate
		self.numSamples = numSamples
		self.returnDataset = returnDataset
		self.batchSize = batchSize
		self.batch = 0
		self.audioColumn = audioColumn
		self.newAudioColumn = newAudioColumn
		self.audioDirPath = audioDirPath
		if bucketColumn is None:
			self.bucketColumn = "bucket"
			self.bucketize(shuffle = shuffle)
		else:
			if (bucketColumn in self.sampleCsv.columns):
				self.bucketColumn = bucketColumn
			else:
				raise Exception(f"Bucket column {bucketColumn} does not exist")
		if (not self.returnDataset) and (self.targetColumn is None):
			raise Exception("Must have target column if not returning entire dataset")

	def bucketize(self, shuffle: bool = True):
		if shuffle:
			self.sampleCsv = self.sampleCsv.sample(frac = 1).reset_index()
		self.sampleCsv["bucket"] = (self.sampleCsv.index.values // self.batchSize)
	
	def nextBatch(self):
		ds = Dataset.from_pandas(self.sampleCsv[self.sampleCsv[self.bucketColumn] == self.batch])\
			.cast_column(self.audioColumn, Audio(sampling_rate=self.samplingRate))
		
		def cropAudioAndCreateNewColumn(row):
			row[self.newAudioColumn] = row[self.audioColumn]["array"][:self.numSamples]
			return row
		
		ds = ds.map(cropAudioAndCreateNewColumn)

		if self.returnDataset:
			return ds
		else:
			ds = ds.with_format("np")
			return (ds[self.newAudioColumn], ds[self.targetColumn])
		