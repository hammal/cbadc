"""
======================
The Murmann ADC Survey
======================

This notebook demonstrate the main
functionality of the :py:class:`cbadc.fom.MurmannSurvey` convenience class
which lets you interact with Murmann's ADC Survey
using a :py:class:`pandas.DataFrame`.
"""

import cbadc.fom
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 10]  # modify default size of plot

adc_survey = cbadc.fom.MurmannSurvey()


###############################################################################
# -------------
# Printing Data
# -------------
#
# To simply output the whole database we access the internal
# db attribute and use the :py:attr:`pandas.DataFrame.style`
# attribute.

print(adc_survey.db.style)

###############################################################################
# -------------------------------------
# Generating the Standard Illustrations
# -------------------------------------
#
# The :py:class:`cbadc.fom.MurmannSurvey` contains several
# convenience functions to quickly generate the standard figures
# from the ADC survey.

# Plot the energy plot
ax = adc_survey.energy()
# we could at this point manipulate the
# axis object (adding more plots, chainging scalings, setting x- and y-limits, etc.)

# Similarly, we can generate the aperture, Walden FoM vs speed, and Schreier FoM vs speed
# equivalently.
_ = adc_survey.aperture()
_ = adc_survey.walden_vs_speed()
_ = adc_survey.schreier_vs_speed()

plt.show()
