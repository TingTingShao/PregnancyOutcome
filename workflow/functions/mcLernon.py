import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

class McLernon:

    def _p3(self, x):
        """(max(x, 0))^3, vectorized"""
        return np.maximum(x, 0) ** 3
    
    def _age_basis_pre(self, age):
        A1=age-25
        A2=(
            self._p3(age-31)
            - (0.59375 * self._p3(age-18))
            - (0.406250 * self._p3(age-50)
            + 203.6563)
        )
        A3 = (self._p3(age - 35)
              - 0.46875   * self._p3(age - 18)
              - 0.53125   * self._p3(age - 50) + 160.78125)
        
        A4 = (self._p3(age - 39)
              - 0.34375   * self._p3(age - 18)
              - 0.65625   * self._p3(age - 50) + 117.90625)
        
        return A1, A2, A3, A4

    def _bmi_basis_pre(self, bmi):
        # knots: 16, 21.7, 24.399, 28.799, 50
        B1 = bmi - 25
        B2 = (self._p3(bmi - 21.7)
              - 0.83235294 * self._p3(bmi - 16.0)
              - 0.16764706 * self._p3(bmi - 50.0) + 570.84833)
        B3 = (self._p3(bmi - 24.399)
              - 0.75294119 * self._p3(bmi - 16.0)
              - 0.24705881 * self._p3(bmi - 50.0) + 548.67810)
        B4 = (self._p3(bmi - 28.799)
              - 0.62352943 * self._p3(bmi - 16.0)
              - 0.37647057 * self._p3(bmi - 50.0) + 454.55295)
        return B1, B2, B3, B4

    def _amh_basis(self, amh):
        # knots: 0.01, 0.98, 2.0999, 4.03, 16
        M1 = amh - 2.5
        M2 = (self._p3(amh - 0.98)
              - 0.93933707 * self._p3(amh - 0.01)
              - 0.06066293 * self._p3(amh - 16.0) + 10.989912)
        M3 = (self._p3(amh - 2.0999)
              - 0.86929959 * self._p3(amh - 0.01)
              - 0.13070041 * self._p3(amh - 16.0) + 13.356415)
        M4 = (self._p3(amh - 4.03)
              - 0.74859285 * self._p3(amh - 0.01)
              - 0.25140715 * self._p3(amh - 16.0) + 11.556963)
        return M1, M2, M3, M4
    
    def equation_pre(
            self, age, bmi, amh, FullTermBirths, MaleInfertility, POCS, Uterine, Unexplained, OvulDisorder
    ):
        A1, A2, A3, A4 = self._age_basis_pre(age)
        B1, B2, B3, B4 = self._bmi_basis_pre(bmi)
        M1, M2, M3, M4 = self._amh_basis(amh)

        FTB = np.asarray(FullTermBirths, dtype=float)
        MI  = np.asarray(MaleInfertility, dtype=float)
        PC  = np.asarray(POCS,       dtype=float)
        UT  = np.asarray(Uterine,         dtype=float)
        UE  = np.asarray(Unexplained,     dtype=float)
        OD  = np.asarray(OvulDisorder,    dtype=float)

        XB = (
            0.4346214
            + 0.0920238  * A1 + 0.0011043  * A2 + (-0.0039663) * A3 + 0.0042808 * A4
            + 0.0664307  * B1 + 0.0019531  * B2 + (-0.0012456) * B3 + (-0.0000662) * B4
            + 1.095414   * M1 + 0.234447   * M2 + (-0.0890884) * M3 + (-0.009418)  * M4
            + 0.0495487  * FTB + 0.0803214 * MI
            + 0.0349664  * PC  + (-0.1772406) * UT + 0.0656224 * UE
            + (-0.1695679) * OD
        )
        return XB

    def predict_pre(
        self,
        age, bmi, amh,
        FullTermBirths, MaleInfertility, polycpcos, Uterine, Unexplained, OvulDisorder,
        return_dict: bool = False
        ):
        """PCycle1/2/3 and cumulative CumPCycle1/2/3 for the Pre-treatment model."""
        XB = self.equation_pre(age, bmi, amh, FullTermBirths, MaleInfertility, polycpcos, Uterine, Unexplained, OvulDisorder)

        PCycle1 = expit(XB)
        PCycle2 = expit(XB - 0.4993235)
        PCycle3 = expit(XB - 0.7894117)

        CumPCycle1 = 1 - (1 - PCycle1)
        CumPCycle2 = 1 - ((1 - PCycle1) * (1 - PCycle2))
        CumPCycle3 = 1 - ((1 - PCycle1) * (1 - PCycle2) * (1 - PCycle3))
        
        out = {
            "XB_pre": XB,
            "PCycle1": PCycle1,
            "PCycle2": PCycle2,
            "PCycle3": PCycle3,
            "CumPCycle1": CumPCycle1,
            "CumPCycle2": CumPCycle2,
            "CumPCycle3": CumPCycle3,
        }
        return out if return_dict else tuple(out[k] for k in out)

# ------- simplified functions for pre treatment model -------
    def construct_dataset_pre(
            self, df
    ):
        A1, A2, A3, A4 = self._age_basis_pre(df['age'])
        B1, B2, B3, B4 = self._bmi_basis_pre(df['bmi'])
        M1, M2, M3, M4 = self._amh_basis(df['amh'])
        OD  = np.asarray(df['OvulDisorder'],    dtype=float)
        # return a dataframe with engineered features then fit with logistic regression
        df_reframed=pd.DataFrame({
            "A1": A1, "A2": A2, "A3": A3, "A4": A4,
            "B1": B1, "B2": B2, "B3": B3, "B4": B4,
            "M1": M1, "M2": M2, "M3": M3, "M4": M4,
            "OD": OD
        })
        # model=LogisticRegression(penalty=None,fit_intercept=True, max_iter=1000)
        # model.fit(df, y_true)

        # XB=(
        #     model.intercept_[0]
        #     + model.coef_[0][0] * A1 + model.coef_[0][1] * A2 + model.coef_[0][2] * A3 + model.coef_[0][3] * A4
        #     + model.coef_[0][4] * B1 + model.coef_[0][5] * B2 + model.coef_[0][6] * B3 + model.coef_[0][7] * B4
        #     + model.coef_[0][8] * M1 + model.coef_[0][9] * M2 + model.coef_[0][10] * M3 + model.coef_[0][11] * M4
        #     + model.coef_[0][12] * OD
        # )
        return df_reframed
    
    def equation_pre_simplified(
            self, age, bmi, amh, OvulDisorder, y_true
    ):
        A1, A2, A3, A4 = self._age_basis_pre(age)
        B1, B2, B3, B4 = self._bmi_basis_pre(bmi)
        M1, M2, M3, M4 = self._amh_basis(amh)
        OD  = np.asarray(OvulDisorder,    dtype=float)
        # return a dataframe with engineered features then fit with logistic regression
        df=pd.DataFrame({
            "A1": A1, "A2": A2, "A3": A3, "A4": A4,
            "B1": B1, "B2": B2, "B3": B3, "B4": B4,
            "M1": M1, "M2": M2, "M3": M3, "M4": M4,
            "OD": OD
        })
        model=LogisticRegression(penalty=None,fit_intercept=True, max_iter=1000)
        model.fit(df, y_true)

        # XB=(
        #     model.intercept_[0]
        #     + model.coef_[0][0] * A1 + model.coef_[0][1] * A2 + model.coef_[0][2] * A3 + model.coef_[0][3] * A4
        #     + model.coef_[0][4] * B1 + model.coef_[0][5] * B2 + model.coef_[0][6] * B3 + model.coef_[0][7] * B4
        #     + model.coef_[0][8] * M1 + model.coef_[0][9] * M2 + model.coef_[0][10] * M3 + model.coef_[0][11] * M4
        #     + model.coef_[0][12] * OD
        # )
        return model
    def get_XB_pre_simplified(
        self, model, age, bmi, amh, OvulDisorder
        ):
        A1, A2, A3, A4 = self._age_basis_pre(age)
        B1, B2, B3, B4 = self._bmi_basis_pre(bmi)
        M1, M2, M3, M4 = self._amh_basis(amh)
        OD  = np.asarray(OvulDisorder,    dtype=float)
        XB=(
            model.intercept_[0]
            + model.coef_[0][0] * A1 + model.coef_[0][1] * A2 + model.coef_[0][2] * A3 + model.coef_[0][3] * A4
            + model.coef_[0][4] * B1 + model.coef_[0][5] * B2 + model.coef_[0][6] * B3 + model.coef_[0][7] * B4
            + model.coef_[0][8] * M1 + model.coef_[0][9] * M2 + model.coef_[0][10] * M3 + model.coef_[0][11] * M4
            + model.coef_[0][12] * OD
        )
        return XB
    
    def predict_pre_simplified(
        self,
        XB,
        return_dict: bool = False
        ):
        """PCycle1/2/3 and cumulative CumPCycle1/2/3 for the Pre-treatment model."""
        # XB = self.equation_pre_simplified(age, bmi, amh, OvulDisorder, y_true)

        PCycle1 = expit(XB)
        PCycle2 = expit(XB - 0.4993235)
        PCycle3 = expit(XB - 0.7894117)

        CumPCycle1 = 1 - (1 - PCycle1)
        CumPCycle2 = 1 - ((1 - PCycle1) * (1 - PCycle2))
        CumPCycle3 = 1 - ((1 - PCycle1) * (1 - PCycle2) * (1 - PCycle3))
        
        out = {
            "XB_pre_cimplied": XB,
            "PCycle1": PCycle1,
            "PCycle2": PCycle2,
            "PCycle3": PCycle3,
            "CumPCycle1": CumPCycle1,
            "CumPCycle2": CumPCycle2,
            "CumPCycle3": CumPCycle3,
        }
        return out if return_dict else tuple(out[k] for k in out)  

# ------- post treatment model -------

    def _age_basis_post(self, age):
        A1 = age - 25
        A2 = (self._p3(age - 33)
              - 0.5666666 * self._p3(age - 20)
              - 0.43333334 * self._p3(age - 50) + 70.833336)
        A3 = (self._p3(age - 37)
              - 0.43333334 * self._p3(age - 20)
              - 0.5666660  * self._p3(age - 50) + 54.166668)
        A4 = (self._p3(age - 40)
              - 0.33333334 * self._p3(age - 20)
              - 0.6666666  * self._p3(age - 50) + 41.666668)
        return A1, A2, A3, A4

    def _bmi_basis_post(self, bmi):
        bmi1 = bmi - 25
        bmi2 = self._p3(bmi - 21.7) - 0.83679527 * self._p3(bmi - 16.2) - 0.16320473 * self._p3(bmi - 49.9) + 534.31543
        bmi3 = self._p3(bmi - 24.5) - 0.75370926 * self._p3(bmi - 16.2) - 0.24629074 * self._p3(bmi - 49.9) + 513.50659
        bmi4 = self._p3(bmi - 29.0) - 0.62017804 * self._p3(bmi - 16.2) - 0.37982196 * self._p3(bmi - 49.9) + 422.63388
        return bmi1, bmi2, bmi3, bmi4

    def _retr_basis(self, retr):
        R1 = retr - 9
        R2 = (self._p3(retr - 2)
              - 0.93103451 * self._p3(retr - 0)
              - 0.06896549 * self._p3(retr - 29) + 335.72415)
        R3 = (self._p3(retr - 7)
              - 0.75862068 * self._p3(retr - 0)
              - 0.24137932 * self._p3(retr - 29) + 545.03448)
        R4 = (self._p3(retr - 13)
              - 0.58620691 * self._p3(retr - 0)
              - 0.41379309 * self._p3(retr - 29) + 427.34482)
        return R1, R2, R3, R4

    def linear_predictor_post(
        self,
        age, bmi, retr, MaleInfertility, OvulDisorder, polycpcos, Uterine,
    ):
        """XB for Post-treatment model."""
        A1, A2, A3, A4 = self._age_basis_post(age)
        B1, B2, B3, B4 = self._bmi_basis_post(bmi)
        R1, R2, R3, R4 = self._retr_basis(retr)

        MI = np.asarray(MaleInfertility, dtype=float)
        OD = np.asarray(OvulDisorder,   dtype=float)
        PC = np.asarray(polycpcos,      dtype=float)
        UT = np.asarray(Uterine,        dtype=float)

        XB = (
            -0.1404629
            + 0.0307055  * A1 + (-0.0002680) * A2 + (-0.0003146) * A3 + 0.0013544 * A4
            + 0.0828801  * B1 +  0.0019979  * B2 + (-0.0011626) * B3 + (-0.0001167) * B4
            + (-0.0697645) * R1 + (-0.0072497) * R2 +  0.0031553  * R3 + (-0.0005503) * R4
            + 0.1618111 * MI + (-0.4128434) * OD + 0.1345618 * PC + (-0.2858210) * UT
        )
        return XB
    def predict_post(
        self,
        age, bmi, retr, MaleInfertility, OvulDisorder, polycpcos, Uterine,
        return_dict: bool = False,
    ):
        """PCycle2/3 and cumulative CumPCycle2/3 for the Post-treatment model."""
        XB = self.linear_predictor_post(age, bmi, retr, MaleInfertility, OvulDisorder, polycpcos, Uterine)

        PCycle2 = expit(XB)
        PCycle3 = expit(XB - 0.3670816)

        CumPCycle2 = 1 - (1 - PCycle2)
        CumPCycle3 = 1 - ((1 - PCycle2) * (1 - PCycle3))

        out = {
            "XB_post": XB,
            "PCycle2": PCycle2,
            "PCycle3": PCycle3,
            "CumPCycle2": CumPCycle2,
            "CumPCycle3": CumPCycle3,
        }
        return out if return_dict else tuple(out[k] for k in out)
    
# ------- post treatment simplified mclernon model -------  
    def construct_dataset_post(
            self, df
    ):
        A1, A2, A3, A4 = self._age_basis_post(df['age'])
        B1, B2, B3, B4 = self._bmi_basis_post(df['bmi'])
        R1, R2, R3, R4 = self._retr_basis(df['retr'])
        OD = np.asarray(df['OvulDisorder'],   dtype=float)

        df=pd.DataFrame({
            "A1": A1, "A2": A2, "A3": A3, "A4": A4,
            "B1": B1, "B2": B2, "B3": B3, "B4": B4,
            "R1": R1, "R2": R2, "R3": R3, "R4": R4,
            "OD": OD
        })     
        return df
      
    def linear_predictor_post_simplified(
        self,
        age, bmi, retr, OvulDisorder,
        y_true
    ):
        """XB for Post-treatment model."""
        A1, A2, A3, A4 = self._age_basis_post(age)
        B1, B2, B3, B4 = self._bmi_basis_post(bmi)
        R1, R2, R3, R4 = self._retr_basis(retr)
        OD = np.asarray(OvulDisorder,   dtype=float)

        df=pd.DataFrame({
            "A1": A1, "A2": A2, "A3": A3, "A4": A4,
            "B1": B1, "B2": B2, "B3": B3, "B4": B4,
            "R1": R1, "R2": R2, "R3": R3, "R4": R4,
            "OD": OD
        })
        model=LogisticRegression(penalty=None,fit_intercept=True, max_iter=1000)
        model.fit(df, y_true)
        # XB=(
        #     model.intercept_[0]
        #     + model.coef_[0][0] * A1 + model.coef_[0][1] * A2 + model.coef_[0][2] * A3 + model.coef_[0][3] * A4
        #     + model.coef_[0][4] * B1 + model.coef_[0][5] * B2 + model.coef_[0][6] * B3 + model.coef_[0][7] * B4
        #     + model.coef_[0][8] * R1 + model.coef_[0][9] * R2 + model.coef_[0][10] * R3 + model.coef_[0][11] * R4
        #     + model.coef_[0][12] * OD
        # )
        return model

    def get_XB_post_simplified(
        self, model, age, bmi, retr, OvulDisorder
        ):
        A1, A2, A3, A4 = self._age_basis_post(age)
        B1, B2, B3, B4 = self._bmi_basis_post(bmi)
        R1, R2, R3, R4 = self._retr_basis(retr)
        OD = np.asarray(OvulDisorder,   dtype=float)
        XB=(
            model.intercept_[0]
            + model.coef_[0][0] * A1 + model.coef_[0][1] * A2 + model.coef_[0][2] * A3 + model.coef_[0][3] * A4
            + model.coef_[0][4] * B1 + model.coef_[0][5] * B2 + model.coef_[0][6] * B3 + model.coef_[0][7] * B4
            + model.coef_[0][8] * R1 + model.coef_[0][9] * R2 + model.coef_[0][10] * R3 + model.coef_[0][11] * R4
            + model.coef_[0][12] * OD
        )
        return XB
    
    def predict_post_simplified(
        self,
        XB,
        return_dict: bool = False,
    ):
        """PCycle2/3 and cumulative CumPCycle2/3 for the Post-treatment model."""
        # XB = self.linear_predictor_post_simplified(age, bmi, retr, OvulDisorder, y_true)

        PCycle2 = expit(XB)
        PCycle3 = expit(XB - 0.3670816)

        CumPCycle2 = 1 - (1 - PCycle2)
        CumPCycle3 = 1 - ((1 - PCycle2) * (1 - PCycle3))

        out = {
            "XB_post": XB,
            "PCycle2": PCycle2,
            "PCycle3": PCycle3,
            "CumPCycle2": CumPCycle2,
            "CumPCycle3": CumPCycle3,
        }
        return out if return_dict else tuple(out[k] for k in out)

    # -------- DataFrame helpers --------
    def predict_pre_df(self, df: pd.DataFrame, colmap: dict | None = None, add_columns=True) -> pd.DataFrame:
        if colmap is None:
            colmap = {
                "age": "age", "bmi": "bmi", "amh": "amh",
                "FullTermBirths": "FullTermBirths", "MaleInfertility": "MaleInfertility",
                "polycpcos": "polycpcos", "Uterine": "Uterine",
                "Unexplained": "Unexplained", "OvulDisorder": "OvulDisorder",
            }
        out = self.predict_pre(
            df[colmap["age"]], df[colmap["bmi"]], df[colmap["amh"]],
            df[colmap["FullTermBirths"]], df[colmap["MaleInfertility"]],
            df[colmap["polycpcos"]], df[colmap["Uterine"]],
            df[colmap["Unexplained"]], df[colmap["OvulDisorder"]],
            return_dict=True,
        )
        return df.assign(**out) if add_columns else pd.DataFrame(out, index=df.index)

    def predict_pre_simplified_df(self, df_tr: pd.DataFrame, df_te: pd.DataFrame, colmap: dict | None = None, add_columns=True, y_true=None) -> pd.DataFrame:
        if colmap is None:
            colmap = {
                "age": "age", "bmi": "bmi", "amh": "amh",
                "OvulDisorder": "OvulDisorder",
            }
        model = self.equation_pre_simplified(
            df_tr[colmap["age"]], df_tr[colmap["bmi"]], df_tr[colmap["amh"]],
            df_tr[colmap["OvulDisorder"]],
            y_true
        )
        XB=self.get_XB_pre_simplified(
            model,
            df_te[colmap["age"]], df_te[colmap["bmi"]], df_te[colmap["amh"]],
            df_te[colmap["OvulDisorder"]]
        )
        out = self.predict_pre_simplified(
            XB,
            return_dict=True,
        )
        return df_te.assign(**out) if add_columns else pd.DataFrame(out, index=df_te.index)

# ----- post helpers -----
    def predict_post_df(self, df: pd.DataFrame, colmap: dict | None = None, add_columns=True) -> pd.DataFrame:
        if colmap is None:
            colmap = {
                "age": "age", "bmi": "bmi", "retr": "retr",
                "MaleInfertility": "MaleInfertility", "OvulDisorder": "OvulDisorder",
                "polycpcos": "polycpcos", "Uterine": "Uterine",
            }
        out = self.predict_post(
            df[colmap["age"]], df[colmap["bmi"]], df[colmap["retr"]],
            df[colmap["MaleInfertility"]], df[colmap["OvulDisorder"]],
            df[colmap["polycpcos"]], df[colmap["Uterine"]],
            return_dict=True,
        )
        return df.assign(**out) if add_columns else pd.DataFrame(out, index=df.index)

    def predict_post_simplified_df(self, df_tr: pd.DataFrame, df_te: pd.DataFrame, colmap: dict | None = None, add_columns=True, y_true=None) -> pd.DataFrame:
        if colmap is None:
            colmap = {
                "age": "age", "bmi": "bmi", "retr": "retr",
                "OvulDisorder": "OvulDisorder",
            }
        model=self.linear_predictor_post_simplified(
            df_tr[colmap["age"]], df_tr[colmap["bmi"]], df_tr[colmap["retr"]],
            df_tr[colmap["OvulDisorder"]], y_true
        )
        XB=self.get_XB_post_simplified(
            model,
            df_te[colmap["age"]], df_te[colmap["bmi"]], df_te[colmap["retr"]],
            df_te[colmap["OvulDisorder"]]
        )

        out = self.predict_post_simplified(
            XB,
            return_dict=True
        )
        return df_te.assign(**out) if add_columns else pd.DataFrame(out, index=df_te.index)
