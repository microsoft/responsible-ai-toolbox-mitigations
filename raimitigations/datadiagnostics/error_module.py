
class ErrorModule():
	"""
	An error module is one of the basic building blocks of ActiveDetect.
	This error detector applies to the distinct values of a column.
	It provides four methods predict(domain), getRecordSet(error_domain), availTypes(), desc()
	"""
    # -----------------------------------
	def _predict(self, vals):
		"""
		Predicts and returns a list of the subset of a domain that is potentially
        erroneous.

        :param vals: a list of values to predict errors on;
		"""
		raise NotImplemented("An error module must implement predict.")

    # -----------------------------------
	def get_erroneous_rows_in_col(self, col_vals) -> list:
		"""
		This method maps the erroneous values to particular rows in the column, returning a list of erroneous row indices.

        :param col_vals: aa list of values to predict distribution errors on
	
		:return: a list of erroneous row indices in column;
		:rtype: a list
		"""
		raise NotImplemented("An error module must implement get_erroneous_rows_in_col.")

    # -----------------------------------
	def description(self):
		"""
		All error modules should have a human readable description of what the error is.
		"""
		raise NotImplemented("An error module must implement description.")

    # -----------------------------------
	def get_available_types(self) -> list:
		"""
		Returns a list of data types supported by the error detection module.
		"""
		raise NotImplemented("An error module must implement get_available_types")
