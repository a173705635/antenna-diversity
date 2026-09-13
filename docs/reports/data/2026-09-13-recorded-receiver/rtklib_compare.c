/* Local test harness only. Link unmodified upstream RTKLIB sources separately. */
#include "rtklib.h"

int main(void)
{
    int prn, sf, j;
    unsigned int word;
    double tow, xyz[3], clock, variance;
    unsigned char bytes[30];
    while (scanf("%d %lf", &prn, &tow)==2) {
        eph_t eph={0};
        eph.sat=satno(SYS_GPS,prn);
        for (sf=1;sf<=3;sf++) {
            for (j=0;j<10;j++) {
                if (scanf("%u",&word)!=1) return 2;
                bytes[j*3]=(word>>16)&255;
                bytes[j*3+1]=(word>>8)&255;
                bytes[j*3+2]=word&255;
            }
            if (decode_frame(bytes,&eph,NULL,NULL,NULL,NULL)!=sf) return 3;
        }
        eph2pos(gpst2time(eph.week,tow),&eph,xyz,&clock,&variance);
        printf("{\"prn\":%d,\"tow_s\":%.17g,\"position_m\":[%.17g,%.17g,%.17g],\"clock_s\":%.17g,\"ephemeris\":{",prn,tow,xyz[0],xyz[1],xyz[2],clock);
        printf("\"week_mod1024\":%d,\"health\":%d,\"ura_index\":%d,\"iodc\":%d,\"iode\":%d",eph.week%1024,eph.svh,eph.sva,eph.iodc,eph.iode);
        printf(",\"toe\":%.17g,\"toc\":%.17g,\"tgd\":%.17g,\"af0\":%.17g,\"af1\":%.17g,\"af2\":%.17g",eph.toes,time2gpst(eph.toc,NULL),eph.tgd[0],eph.f0,eph.f1,eph.f2);
        printf(",\"sqrt_a\":%.17g,\"e\":%.17g,\"m0\":%.17g,\"delta_n\":%.17g,\"omega\":%.17g,\"omega0\":%.17g,\"omega_dot\":%.17g,\"i0\":%.17g,\"idot\":%.17g",sqrt(eph.A),eph.e,eph.M0,eph.deln,eph.omg,eph.OMG0,eph.OMGd,eph.i0,eph.idot);
        printf(",\"cuc\":%.17g,\"cus\":%.17g,\"crc\":%.17g,\"crs\":%.17g,\"cic\":%.17g,\"cis\":%.17g}}\n",eph.cuc,eph.cus,eph.crc,eph.crs,eph.cic,eph.cis);
    }
    return 0;
}
