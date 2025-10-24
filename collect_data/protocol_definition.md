# Protocol Definition

When starting the program the following data is sent out back to back over the serial connection:

- Duration of reading all 4 PDs (us) [$T_{0,1,2,3}$] [u8]
- Configured sampling frequency (Hz) [$f$] [u16]
- Length of sample (s) [$T_s$] [u8]
- Corrected sample delay (us) [$T_s' = T_s - T_{0,1,2,3}$] [u32]

From here, sending any byte will result in a reading being done. After this reading being complete, an array of `u16` values will be sent across the serial connection. It will be $4 \times f \times T_s$ values, where the format is $$\left[I_0^{t=0},I_1^{t=0},I_2^{t=0},I_3^{t=0},I_0^{t=1},I_1^{t=1},I_2^{t=1},I_3^{t=1},...,I_0^{t=fT_s},I_1^{t=fT_s},I_2^{t=fT_s},I_3^{t=fT_s}\right]$$
, where $I_i^{t=j} \in \mathbb{N}$ means the photodiode intensity at photodiode $i$ at timestamp $j$. After this, it is ready for a new byte to receive / sample to be taken. For debugging, the onboard LED will be toggled on while a sample is being taken. Data is sent in native endianness, in this case little endian.
