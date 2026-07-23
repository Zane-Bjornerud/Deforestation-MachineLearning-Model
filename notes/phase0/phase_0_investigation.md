# Phase 0 investigation

## Observed training tensor

- Image Shape: torch.Size([2, 18, 256, 256])
- Image dtype: torch.float32
- Image minimum: -22.481521606445312
- Image maximum: 15.254830360412598
- Mask shape: torch.Size([2, 1, 256, 256])
- Mask dtype: torch.float32
- Unique mask values: tensor([0., 1.])
- Training file inspected: train.py
- Date inspected: 23/07/26

## Channel Evidence Table
Channel     Actually_is     Meaning            Evidence
- 0         B2_1            blue light1        printed band_names from metadata
- 1         B3              green light
- 2         NBR_1           normalized burn ratio1
- 3         B12             shortwave infrared 2
- 4         B11             shortwave infrared 1
- 5         NBR             normalized burn ratio
- 6         B3_1            green light1
- 7         NDVI_1          normalized difference vegetation index1
- 8         NDVI            normalized difference vegatation index
- 9         B12_1           shortwave infrared 1-1
- 10        dNBR            change index for NBR
- 11        dNDVI           change index for NDVI
- 12        B4_1            red light1
- 13        B11_1           shortwave infrared 1-1
- 14        B8_1            near-infrared1
- 15        B2              blue light
- 16        B4              red light
- 17        B8              near-infrared
** 1 suffix indicates second time period