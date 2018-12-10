

def frequencies_to_val(freq='hour'):
	"""
	Parameters
	----------
	freq : string, default 'hour'


	Returns
	-------
	frequencies[freq] : string
	The mapping for a pandas function when working with intervals

	"""

	freq = freq.lower().strip()

	frequencies = {
		'15min'	:	'15min', 
		'hour'	:	'H', 
		'day'	:	'D'
		}

	if freq not in frequencies.keys():
		raise NotImplementedError('Value not known.')

	return frequencies[freq]
