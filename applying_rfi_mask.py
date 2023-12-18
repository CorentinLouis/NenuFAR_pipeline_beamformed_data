# -*- coding: utf-8 -*-

import numpy


def applying_rfi_mask(
                        self,
                        rfi_mask,
                        rfilevel: int= 1,
                        threshold: float= 0.7
                        ) -> None :
    """
    Apply a RIF mask to data
    Parameters:
            self:
                ReadFits_data object
            rfi_mask:
                ReadFits_rfimask object
            rfilevel (int):
                Level of RFI mitigation mask to apply (between 0 and 3)
            threshold (float):
                useful if rfilevel=0, otherwise not use.
                everything > threshold is set to 1, < threshold set to 0
    """
    if rfilevel == 0:
        mask = rfi_mask.rfimask_level0
        mask [mask>=threshold] = 1
        mask [mask<threshold] = 0
    if rfilevel == 1:
        mask = rfi_mask.rfimask_level1
    if rfilevel == 2:
        mask = rfi_mask.rfimask_level2
    if rfilevel == 3:
        mask = rfi_mask.rfimask_level3

    self.data_masked = numpy.zeros(self.data.shape)+numpy.nan
    for i_stokes in range(self.data.shape[2]):
        self.data_masked[:,:,i_stokes] = self.data[:,:,i_stokes] * mask