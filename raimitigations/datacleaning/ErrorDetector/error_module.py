"""
An error module is one of the basic building blocks of ActiveDetect.
This error detector applies to the distinct values of a column.
It provides four methods predict(domain), getRecordSet(error_domain), availTypes(), desc()
"""

class ErrorModule():
    # -----------------------------------
	def predict(self, vals):
		"""
		Predicts and returns a list of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict errors on;
		"""
		raise NotImplemented("An error module must implement predict.")

    # -----------------------------------
	def get_erroneous_rows_in_col(self, erroneous_vals, dataset, col):
		"""
		This method maps the erroneous values to particular rows in the column, returning a list of erroneous row indices.

        :param erroneous_vals: a list of errors predicted by the predict function;

        :param dataset: dataset containing the column of data evaluated for errors;

        :param col: name or index of column that has been evaluated for errors;
		"""
		raise NotImplemented("An error module must implement get_erroneous_rows_in_col.")

    # -----------------------------------
	def description(self):
		"""
		All error modules should have a human readable description of what the error is.
		"""
		raise NotImplemented("An error module must implement description.")

    # -----------------------------------
	def get_available_types(self):
		"""
		Returns a list of data types supported by the error detection module.
		"""
		raise NotImplemented("An error module must implement get_available_types")
