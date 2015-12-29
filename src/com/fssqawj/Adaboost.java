package com.fssqawj;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;

/**
 * Created by fssqawj on 2015/12/29.
 */
public class Adaboost {

    public static List<Double> spValue = new ArrayList<Double>();
    public static List<Double> erValue = new ArrayList<Double>();
    public static List<Double> sgValue = new ArrayList<Double>();

    public static void main(String[] args) throws Exception {
        // 6 10 14
        for(int i = 1;i < 200;i ++) {
            System.out.print(i + " ");
            train("train.txt", i, 10);
            test("test.txt", 10);
        }
    }

    public static void train(String trainFile, int num, int f) throws IOException {
        spValue.clear();
        erValue.clear();
        sgValue.clear();
        //ReadFile rd = new ReadFile(trainFile);
        BufferedReader rd = null;
        File inFile = new File(trainFile);
        try {
            rd = new BufferedReader(new FileReader(inFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        List<Double> number = new ArrayList<Double>();
        List<Double> tag = new ArrayList<Double>();
        List<Double> weight = new ArrayList<Double>();

        String temp = "";
        int cnt = 0;
        while((temp = rd.readLine()) != null){
            //System.out.println(temp);
            String[] temps = temp.split(",");
            number.add(Double.parseDouble(temps[f])); //3
            tag.add(Double.parseDouble(temps[21])); //21
            //System.out.println(cnt ++);
        }
        rd.close();
        for(int i = 0;i < 299;i ++){
            weight.add(1.0/299);
        }
        //System.out.println(weight.size());
        //System.out.println(tag.size());
        //System.out.println(number.size());
        int class_number = num;
        while((class_number --) > 0){
            List<Double> res = err(number, tag, weight);
            //System.out.println(res);
            double sp = res.get(0);
            spValue.add(sp);
            double errvalue = Math.log((1.0 - res.get(1)) / res.get(1)) / 2;

            sgValue.add(res.get(2));
            //System.out.println("a: " + errvalue);
            erValue.add(errvalue);
            double z = zm(weight, number, tag, errvalue, res.get(0), res.get(2));
            //System.out.println(z);

            for(int i = 0;i < weight.size();i ++){
                if(right(number.get(i), sp, tag.get(i), res.get(2)) == 1.0) {
                    weight.set(i, weight.get(i) * Math.exp(-1.0 * errvalue) / z);
                }
                else weight.set(i, weight.get(i) * Math.exp(errvalue) / z);
            }
            //System.out.println(weight);
        }
    }

    public static void test(String testFile, int f) throws IOException {
        //ReadFile rd = new ReadFile(testFile);

        BufferedReader rd = null;
        File inFile = new File(testFile);
        try {
            rd = new BufferedReader(new FileReader(inFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        List<Double> number = new ArrayList<Double>();
        List<Double> tag = new ArrayList<Double>();
        String temp = "";
        while((temp = rd.readLine()) != null){
            String[] temps = temp.split(",");
            number.add(Double.parseDouble(temps[f]));
            tag.add(Double.parseDouble(temps[21]));
        }
        int hit = 0;
        for(int i = 0;i < number.size();i ++){
            double key = number.get(i);
            double tv = tag.get(i);
            double res = 0;
            for(int j = 0;j < spValue.size();j ++){
                double sp = spValue.get(j);
                double err = erValue.get(j);
                double sg = sgValue.get(j);
                //System.out.println(sp + " " + err);
                if(key < sp && sg == 1.0 || key > sp && sg == -1.0)res += err;
                else res -= err;
            }
            if(sign(res) == tv)hit ++;
        }

        System.out.println("hit: " + 1.0 * hit / number.size());

    }

    public static double right(double number, double sp, double tag, double sg){
        if(number < sp && sg == tag)return 1.0;
        else return -1.0;
    }

    public static double sign(double x){
        return x > 0 ? 1 : -1;
    }

    public static double zm(List<Double> weight, List<Double> number, List<Double> tag, double err, double sp, double sg){
        double res = 0;
        //int cnt = 0;
        for(int i = 0;i < number.size();i ++){
            double key = number.get(i);
            double tv = tag.get(i);
            double wt = weight.get(i);

            if(right(key, sp, tv, sg) == 1.0){
                res += wt * Math.exp(-1.0 * err);
                //System.out.println(cnt);
                //cnt = cnt + 1;
            }
            else res += wt * Math.exp(err);
        }
        return res;
    }
    public static List<Double> err(List<Double> fea, List<Double> type, List<Double> weight){
        ArrayList<Double> ans=new ArrayList<Double>();
        ArrayList<Double> fea1 = new ArrayList<Double>();
        fea1.addAll(fea);
        Collections.sort(fea1);

        int length=fea.size();
        double lower=0;
        double upper=0;
        double fm=0;
        double fw=1;
        double fa=0;
        for(int i=0;i<fea1.size()-1;i++){
            double mid=(fea1.get(i)+fea1.get(i+1))/2;
            double misx=0;double misy=0;

            for(int j=0;j<fea.size();j++){
                double num=fea.get(j);
                double tag=type.get(j);
                double wgt=weight.get(j);

                if(num<=mid && tag== -1 || num>mid && tag == 1){
                    misx=misx+wgt;
                }
                if(num<=mid && tag== 1 || num>mid && tag == -1){
                    misy=misy+wgt;
                }

            }
            if(misx<fw){
                fw=misx;
                fm=mid;
                fa = 1;
            }
            if(misy<fw){
                fw=misy;
                fm=mid;
                fa = -1;
            }

        }

        ans.add(fm);
        ans.add(fw);
        ans.add(fa);
        return ans;
    }
}
