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

adc_survey = cbadc.fom.MurmannSurvey()

# Get info for the resulting dataset
adc_survey.db.info()


###############################################################################
# -------------
# Printing Data
# -------------
#
# To simply output the whole database we access the internal
# db attribute and use the :py:attr:`pandas.DataFrame.style`
# attribute as ``adc_survey.db.style``
#
# Futhermore, one of the key feature of pandas is that we
# can easily search through our data. For example we can
# isolate all publications with an FoMS_hf >= 180 dB
# by commands of the following style.

# get all column names
print(f"Columns: {adc_survey.columns()}")

only_185dB_FoMS = adc_survey.db[adc_survey.db['FOMS_hf [dB]'] >= 180]

# It's also possible to output these (sorted and with selected columns) in text format as

only_185dB_FoMS[['FOMS_hf [dB]', 'AUTHORS', 'TITLE', 'YEAR', 'CONFERENCE']].sort_values(
    'FOMS_hf [dB]', ascending=False
).style.format(precision=1)

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


###############################################################################
# ---------------------------------------------------------------
# Plotting and Extracting Within ENOB and Nyquist frequency range
# ---------------------------------------------------------------
#
# We can also use the :py:class:`cbadc.fom.MurmannSurvey.awht` to
# quickly plot and extract relevant publications within a bandwidth
# and ENOB range


bw = (5e5, 1e7)
enob = (11, 13)
selected_publications = adc_survey.select_bw_and_enob(bw, enob).sort_values(
    'P/fsnyq [pJ]', ascending=True
)

# Make a scatter plot of area vs power
ax = selected_publications.plot.scatter('P [W]', 'AREA [mm^2]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title("Area vs Power for ENOB=[11,13) and BW=[0.5MHz, 10MHz)")
ax.grid(True, which="both")

# Print some attributes of the selected subset
selected_publications[
    [
        'P/fsnyq [pJ]',
        'SNR [dB]',
        'fsnyq [Hz]',
        'P [W]',
        'ARCHITECTURE',
        'AUTHORS',
        'TITLE',
        'YEAR',
    ]
].style.format(precision=1)

# sphinx_gallery_thumbnail_number = 4
