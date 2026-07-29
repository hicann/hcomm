/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <securec.h>

static char g_end_ca[] = {"-----BEGIN CERTIFICATE-----\n\
MIIDizCCAnOgAwIBAgIIffhsme9FaxYwDQYJKoZIhvcNAQELBQAwQjELMAkGA1UE\n\
BhMCQ04xDzANBgNVBAoTBkh1YXdlaTEiMCAGA1UEAxMZSHVhd2VpIDIwMTIgbGFi\n\
b3JhdG9yeSBDQTAeFw0xOTExMTEwMjEzNTJaFw0yMDExMTAwMjEzNTJaMDcxCzAJ\n\
BgNVBAYTAkNOMQ8wDQYDVQQKEwZIdWF3ZWkxFzAVBgNVBAMTDkhpLmh1YXdlaS4x\n\
OTgwMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA/FbiM9Qfb/Ze+dlM\n\
dBpZoM0GJ8O+L0NaVxSBXRTPcXAtdVQMkJyprRK8RnZxeaDhbui8TXDO14aF0IN+\n\
BOwLjS/WsUg/SidtHLYSurKGerIBSUHJGtTARlpwlrOf50TJ3xd8P+/w+2Jq4gq9\n\
PWpFQUeJ/WnRA2BUFBJH0ssilU/W+FKyt1kWS2nYq2U9hTRJmTx58ac8EHspxOuS\n\
bjU9nPa1mvAkqH23ALCPdMtFEfrhN30VV1K+rFV4xVblzGQckckQGWGOnBjkgqBH\n\
XAhok0gvu9ZnCqTFV9VGm4GCBqP+UCEHvuAwVFDcQ+mRq6kvAvOf9qWF9IAN90jp\n\
4JwqMwIDAQABo4GPMIGMMB8GA1UdIwQYMBaAFHOfx1/hlqgOeXF53GnLCvG84PTl\n\
MAsGA1UdDwQEAwID+DBcBggrBgEFBQcBAQRQME4wKAYIKwYBBQUHMAKGHGh0dHA6\n\
Ly8xMjcuMC4wLjEvY2Fpc3N1ZS5odG0wIgYIKwYBBQUHMAGGFmh0dHA6Ly8xMjcu\n\
MC4wLjE6MjA0NDMwDQYJKoZIhvcNAQELBQADggEBADLfMHYGmdAukXLaS7qW/DWJ\n\
xt52kD40t6VCGaSG0JDXnrWaWQyjhTFlzDCM9lJqk5ozwNZTmkNsM79Hf8OIpb4l\n\
QvOsDSAXr43jVWG5ohCWFaNBR9C23G6fEPe8fuyoX8Exz2ga1UTPQXPoTgycYDnd\n\
i+gkYu+qz47U/bYyp0a32bbunyaOKSLzCRKHgnTB3myKOYM6p5pTb8ftUCT4HKyK\n\
P2/w4W2Y+oEdSz8Diep1IelfOu9NCWnsbVIVgDZit3L5hFi9FWTj/XU3z5gACPV3\n\
peMyQA8cuwyoeHYhcYZT/gM2IQAZaBWEjv+uEF87eRnEGsJ3zmRDjl/ZGwXfWjQ=\n\
-----END CERTIFICATE-----"};

static char g_root_ca[] = {"-----BEGIN CERTIFICATE-----\n\
MIIEtjCCAp6gAwIBAgIRdkeE+a6EVyCmTfYyjUiOjlwwDQYJKoZIhvcNAQELBQAw\n\
PDELMAkGA1UEBhMCQ04xDzANBgNVBAoTBkh1YXdlaTEcMBoGA1UEAxMTSHVhd2Vp\n\
IEVxdWlwbWVudCBDQTAeFw0xNzExMjMwMTIxMDNaFw00MTExMTIwMTIxMDNaMEIx\n\
CzAJBgNVBAYTAkNOMQ8wDQYDVQQKEwZIdWF3ZWkxIjAgBgNVBAMTGUh1YXdlaSAy\n\
MDEyIGxhYm9yYXRvcnkgQ0EwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIB\n\
AQCsF89N/u/J34AY7iX3nLvdX3To6aL6apDVzCOAkhaGScOV4m/kEh1YOuPGUlb9\n\
Umv46oBag6d5jDYuiLqaATxacz0jOn/LWYGiW7PRd47VVk0L4ucL/rVWmm3c0pAq\n\
J1ZDrc6JVHYtZct9eW2RFARiPXDpjMeQ9Mq25R+QdoZN3EC85K5I189SFd0cRraU\n\
3XGBBjH2jMaVoFtxOBrx+vZu+ccdzQ/PCzROdj2MM7XnviEmkqdoylW88px/niki\n\
m975Ij1/DhpVh3O74GDrPBw9GYKobx3zKxDnU9d2qD/WY5caGtZQRnXKDcTQbj7d\n\
1p/oE2ukvhVBeGQdVbmV4WzxAgMBAAGjgawwgakwHwYDVR0jBBgwFoAUKvgQWSeA\n\
NR+nfLo7nyrkSqqbkuowRgYDVR0gBD8wPTA7BgRVHSAAMDMwMQYIKwYBBQUHAgEW\n\
JWh0dHA6Ly9zdXBwb3J0Lmh1YXdlaS5jb20vc3VwcG9ydC9wa2kwDwYDVR0TBAgw\n\
BgEB/wIBADAOBgNVHQ8BAf8EBAMCAQYwHQYDVR0OBBYEFHOfx1/hlqgOeXF53GnL\n\
CvG84PTlMA0GCSqGSIb3DQEBCwUAA4ICAQAZDpkbFNYFVu/bi/x1w9scTKx3fYXO\n\
5DuoaEaRCsNTEAgcFmUsh8CFU1ORi9NSI5j91CgVUSBp848hVkJmuXaxnxZOrjWc\n\
QmhfbLcny2TSR0Ffgi8zXNsqYvQVIp81tKzCQnT/3EDeSEVu6jKe2twwhBFkGGpc\n\
BKxlIq3BFrBMaJnI7kNGW0ea5bWm98xD5rGS5ATNYaaUVI1B0MxzDGg3IfNxpKU2\n\
nTnYMkmuqpr2qdEM9kbMyWiW/TRpUODzase3KiE+9fb2jITH2ta9qKedBAdRExBg\n\
2c9MoTSYLA3NOUg6LjXOB/+wUDr5YFNxLGJRo5zAxWjTDoddIZJ4RR6LGcNuIJl9\n\
ykG3vzVWmNJ+6BpLzkJ2tDP7hsSeWzs83+L14VxMAhRO+UUulwTgaHu5vBBhkZpG\n\
6qVq3f+smtOOOULDCn9mLKKNOhfvL74YbV9H6m9wnzzcIOCSfZ1ev5x0D00oXJMb\n\
WigP+znyT/1jb46M0n6IdZQZ5xK7x11bWQnBPfWEzW4cWDwhkEMfXr+sCToCQ0+U\n\
t2f37SGLILtM9mOrz5CxaIysvPHbqzdKu9obT34maYWnX+FoFwfyJErix4sqaj56\n\
77daULe75H/LtYCkbTXE+LzG5I7lPqg7NhK4gApkuj5RJT/mn2PxRg2z66eclseZ\n\
JRJoN9BBmLJ+Nw==\n\
-----END CERTIFICATE-----"};

static char g_snd_ca[] = {"-----BEGIN CERTIFICATE-----\n\
MIIFPzCCAyegAwIBAgIRdiVgAipHXuymK3xorJs/KYYwDQYJKoZIhvcNAQELBQAw\n\
PDELMAkGA1UEBhMCQ04xDzANBgNVBAoTBkh1YXdlaTEcMBoGA1UEAxMTSHVhd2Vp\n\
IEVxdWlwbWVudCBDQTAeFw0xMTEyMDYwNzM0MjNaFw00MTExMjgwNzM0MjNaMDwx\n\
CzAJBgNVBAYTAkNOMQ8wDQYDVQQKEwZIdWF3ZWkxHDAaBgNVBAMTE0h1YXdlaSBF\n\
cXVpcG1lbnQgQ0EwggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQCiiYQn\n\
C/Mp9obmAnXmu/Nj6rccSkEQJXlZipOv8tIjvr0B8ObpFUnU+qLojZUYlNmXH8Rg\n\
RgFB1sBSyOuGiiP0uNtJ0lPLbylsc+2fr2Rlt/qbYs1oQGz+oNl+UdAOtm/lPzgg\n\
UOVVst15Ovf0Yf6LQ3CQalN2VJWgKpFUudDKWQ2fzbFT5YSfvhFxvtvWfgdntKAJ\n\
t3sFvkKr9Qw+0EYNpQiw5EALeLWCZSYU7A939puqYR6aNA447S1K8SgWoav82P4U\n\
Y/ykLXjcgTeCnvRRtUga1gdIwm5d/vRlB5il5wspGLLes4SomzUYrvnvHio555NZ\n\
PpvmpIXNolwvYW5opAyYzE05pVSOmHf/RY/dHto8XWexOJq/UAFBMyiH4NT4cZpW\n\
jYWR7W9GxRXApmQrrLXte1CF/IzXWBMA2tSL0WnRJz5HRcKzsOC6FksiqsYstFjc\n\
CE7J7Nicr3Bwq5FrZiqGSdLmLRn97XqVlWdN31HX16fzRhZMiOkvQe+uYT+BXbhU\n\
1fZIh6RRAH3V1APobVlCXh5PDq8Ca4dClHNHYp5RP0Pb5zBowTqBzSv7ssHrNceQ\n\
sWDeNjX9t59NwviaIlXIlPiWEEJc22XtMm4sc/+8mgOFMNXr4FWu8vdG2fgRpeWJ\n\
O0E035D6TClu4So2GlN/fIccp5wVYAWF1WhxSQIDAQABozwwOjAMBgNVHRMEBTAD\n\
AQH/MAsGA1UdDwQEAwIBBjAdBgNVHQ4EFgQUKvgQWSeANR+nfLo7nyrkSqqbkuow\n\
DQYJKoZIhvcNAQELBQADggIBAEDHZJ4vvx2kPmHEsN3OJOeF2nV6chjF1QZcUwlo\n\
jhUtIv9jte9mci5qllvYRU5mia9rYZiP61XfdrwORf8QdJcI63QgrIj7MtnJULcU\n\
Ukk0Sj9Fz6rswfhlaqtRjDp2ljizCl9bmUzKZTl40m/SMbItbSyYXvKrgSPTwgPo\n\
/MralqpJcuoUkf+JDZIP3AaIy+vecksJwmoFIc0OqwP7uNC55kr8kx70eH3QKaiA\n\
U+8CL3N7gtMFBL2MALlk3vFEICEAhWvMGrYNtSzBUEJNTspx+qVxERBqxJImBsPG\n\
D7LhLOaPlSzfbU6CD3C8G92Y7r4nCcQ+SOQv4k6TTRn8pOj5c0oy3Z28DeZGuzSX\n\
NPsWur3aRVwE0mOY8cLBkgio7AQjqIAmdbo5vie7X1zshyEcA7FaE1mJdNS3WVCv\n\
lMwTFwygq13svLQ5MwGPSexsHudZ5JP55tHXkQyPRqxdhFr+gxDw5oiv/LlxApB8\n\
5MwEfTTs/uzS6FSWAUC0IAxWyZ3MytVAAL7SiwZp/eODWBwLXETlIKcu/fdhTfN5\n\
q1Mm9TjMjJmDEoqzIDRjDuVR4v/3czRxMOkKtUHJt2ixeiidh9hjY6ae669BqpBR\n\
W0d5dyNozy+IJcUo7Gg2+F1AhTLwvPiYlJLsNGZZvqXfhplpwcAnvtoGJvAj+QkL\n\
iW4z\n\
-----END CERTIFICATE-----"};

static char g_py[] = {"-----BEGIN RSA PRIVATE KEY-----\n\
MIIEpAIBAAKCAQEA/FbiM9Qfb/Ze+dlMdBpZoM0GJ8O+L0NaVxSBXRTPcXAtdVQM\n\
kJyprRK8RnZxeaDhbui8TXDO14aF0IN+BOwLjS/WsUg/SidtHLYSurKGerIBSUHJ\n\
GtTARlpwlrOf50TJ3xd8P+/w+2Jq4gq9PWpFQUeJ/WnRA2BUFBJH0ssilU/W+FKy\n\
t1kWS2nYq2U9hTRJmTx58ac8EHspxOuSbjU9nPa1mvAkqH23ALCPdMtFEfrhN30V\n\
V1K+rFV4xVblzGQckckQGWGOnBjkgqBHXAhok0gvu9ZnCqTFV9VGm4GCBqP+UCEH\n\
vuAwVFDcQ+mRq6kvAvOf9qWF9IAN90jp4JwqMwIDAQABAoIBAFYc3FLtl9DglebT\n\
thiFCYMrlGcrkJKWfJTtBXybZnOC4bCBf0w2elz64M93CiCEu4na2K/gyGWanb3b\n\
sbzCROrooW6chiSrMbwzkk6uL+BdML0sOxHx+q/Hm1aCdBY7TlzYqekz29wd6PW8\n\
znnP81Rksn2Nh+uMCwVSe7o+4k1rZSsJVRKOfg/KLp/SEb23X0L0bYerrpmB/OEV\n\
5/iv7PuajHrrWh6yM06oX+f5gIZBiMWQ2WEDGHubJjRyswWUyK2rEea2OF1bJT+z\n\
0Zq9TmY/oA2MCb4XUtSL59W6FtZ9Isnc8NHS51XjB061u33thcQ34E2woD3QT0oi\n\
cGtD5JkCgYEA/lHcVL5Rm5hzQUHlfcCszf8KGOaZU7G17DvgK5G8T1JNw1KncN8c\n\
jHZfJhx6GMfrYpEHBD8vv6ri290YteapkYFSe240ZdmsctNEhk3WapYSmjBIvTBD\n\
0C11O5KOb+iYcRupm1Aeg8fLgzgxiEKx+zEAa4EueywAHOlxC2USylUCgYEA/gGs\n\
Z5Dwq75w9fW1zeUjjjXlIgGklfQbDM6SoSyTa/TLQw39NmPhUhblRTdKFPnDUXhN\n\
Tnu0ORx9CvVCNfvSk+svWre+Q9urLNkF0nKayGnQIXnhBlPWGgNeYQZi6FqkmVaU\n\
fWBHqhcfMor66oUZ0evqJ/+lvpRAeEcFehPfumcCgYAux7IGqIdsXot8ynlDO2jN\n\
74bU873qZjr3fEAM457G3HXPYunH2lJvB+sSoJRY8JU8qT6oKlNHJ1DZbn74Hri8\n\
OhSI/cmHnpWY/YGSTskNDBPZ5t3KZxFiPqpczeWDcj5wN81n80HZrauitHhv/wys\n\
DuRr4fRB1eMjblFL7kiZjQKBgQCg+PSGqd8sKEO0TGRMOMPgsx0kAQCKG6os4pkg\n\
VXyT9Q3/z9TB7Gh2OpZP2Cs1wddbQS9U5qafbwN7t1Sfm5inL2vSRRHqNUN055B9\n\
/y1Ch3RkUrYd6XGNCMd+G6sA77jSiIEQN70S+RZHVLaRe4qSc7zwXl5uuctlrjS1\n\
WfT2TwKBgQCO1SMNaA7HL1E6RgV2hJODaVqXSsehgd852RD0yYAAFpdSIFq98c/G\n\
1eLvFmcOojc3GAkFTcuttCJ9IabBg3n6BjMebxxZC4ki9a1fnE/40Tri8QtBf84D\n\
q1MY0x69YkICnsoRuybgeIMZE2XMIIG9CuKrsewb4m4dFCa7Shk86w==\n\
-----END RSA PRIVATE KEY-----"};

#define TLS_MAGIC_WORDS_LEN 8
#define RS_SALT_MAX_LEN 48
#define RS_CERT_COUNT 15
#define TLS_SALT_LEN 48
#define IV_LEN 16
#define BLOCK_KY_LEN 32
#define PWD_MIN_LEN 8
#define PWD_MAX_LEN 15
#define FGETS_MAX_LEN 32
#define PWD_ENC_LEN 256
#define WORK_KEY_LEN 516
#define TAG_LEN 16
#define MAX_CERT_COUNT 15
#define RS_MAGIC_WORDS "1234567"

struct cert_infos {
    char cert_info[2048];
};

struct certs {
    struct cert_infos certs[RS_CERT_COUNT];
};

#define TLS_RES_LEN 1024
struct rs_cert_manage_info {
    char magic_words[TLS_MAGIC_WORDS_LEN]; /* 1234567 */
    unsigned int cert_count; /* num of certs */
    int state; /* 0:not ok 1:ok */
    unsigned int ca_wcout; /* counts of ca writing flash */
    unsigned int cert_ky_wcout; /* counts of eqpt and key writing flash */
    unsigned int crl_wcout; /* counts of crl writing flash */
    unsigned int crl_len; /* len of crl */
    unsigned int ky_len; /* len of key */
    unsigned int ky_enc_len; /* len of enc key */
    unsigned char salt[TLS_SALT_LEN]; /* salt */
    unsigned int salt_size; /* len of salt */
    unsigned int cert_len[MAX_CERT_COUNT];
    unsigned int total_cert_len; /* not include head only len of certs */
    unsigned int tls_enable;
    unsigned int tls_alarm;
    unsigned int pwd_len; /* len of pwd */
    unsigned int pwd_enc_len; /* len of enc pwd */
    unsigned char enc_pwd[PWD_ENC_LEN];
    unsigned int work_key_len; /* len of work_key */
    unsigned char work_key[WORK_KEY_LEN];
    unsigned char iv[IV_LEN]; /* initial vector */
    unsigned int iv_size; /* len of initial vector */
    unsigned char tag[TAG_LEN];
    unsigned int tag_len;
    unsigned int save_mode;
    char res[TLS_RES_LEN];
};

int devdrv_get_user_config(unsigned int dev_id, const char* name, unsigned char* buf, unsigned int *buf_size)
{
    int ret;
    if (strcmp(name, "hccp_certs_mng_cb") == 0) {
        struct rs_cert_manage_info *mng_infos = (struct rs_cert_manage_info *)buf;
        mng_infos->cert_count = 1;
        mng_infos->total_cert_len = strlen(g_end_ca);
        mng_infos->ky_len = strlen(g_py);
        mng_infos->ky_enc_len = strlen(g_py) + 16;
        mng_infos->tls_enable = 0;
        mng_infos->pwd_len = PWD_MAX_LEN;
        mng_infos->pwd_enc_len = PWD_ENC_LEN;
        mng_infos->work_key_len = WORK_KEY_LEN;
        mng_infos->salt_size = TLS_SALT_LEN;
        mng_infos->iv_size = IV_LEN;
        mng_infos->tag_len = TAG_LEN;
        mng_infos->save_mode = 0;
        ret = memcpy_s(mng_infos->magic_words, sizeof(mng_infos->magic_words), "1234567", sizeof("1234567"));
        mng_infos->salt_size = 7;
        return 0;
    } else if (strcmp(name, "hccp_certs_eqpt_cb") == 0) {
        struct certs *certs = (struct certs *)buf;
        ret = memcpy_s(certs->certs[0].cert_info, sizeof(certs->certs[0].cert_info), g_end_ca, strlen(g_end_ca));
        ret = memcpy_s(certs->certs[1].cert_info, sizeof(certs->certs[1].cert_info), g_root_ca, strlen(g_root_ca));
        ret = memcpy_s(certs->certs[2].cert_info, sizeof(certs->certs[2].cert_info), g_snd_ca, strlen(g_snd_ca));
        return 0;
    } else if (strcmp(name, "hccp_pri_data_cb") == 0) {
        ret = memcpy_s(buf, strlen(g_py), g_py, strlen(g_py));
        *buf_size = 5120;
        return 0;
    } else if (strcmp(name, "hccp_certs_revoc_cb") == 0) {
        *buf_size = 40960;
        return -1;
    } else {
        return -1;
    }
}

int tls_get_user_config(unsigned int save_mode, unsigned int chipId, const char *name,
    unsigned char *buf, unsigned int *buf_size)
{
    int ret;

    ret = devdrv_get_user_config(chipId, name, buf, buf_size);

    return ret;
}

void tls_get_enable_info(unsigned int save_mode, unsigned int chipId, unsigned char *buf, unsigned int buf_size)
{
    return;
}

int halSetUserConfig(unsigned int dev_id, const char *name, unsigned char *buf, unsigned int buf_size)
{
    return 0;
}

int halClearUserConfig(unsigned int devid, const char *name)
{
    return 0;
}

int NetCommGetSelfHome(char *userNamePath, unsigned int pathLen)
{
    memcpy(userNamePath, "/tmp", strlen("/tmp"));
    return 0;
}

int get_tls_config_path(char *user_name_path, unsigned int path_len)
{
    return 0;
}

int FileReadCfg(const char *filePath, int devId, const char *confName, char *confValue, unsigned int len)
{
    if (strncmp(confName, "udp_port_mode", strlen("udp_port_mode") + 1) == 0){
        memcpy_s(confValue, len, "nslb_dp", strlen("nslb_dp"));
    } else {
        memcpy_s(confValue, len, "16666", strlen("16666"));
    }
    return 0;
}
