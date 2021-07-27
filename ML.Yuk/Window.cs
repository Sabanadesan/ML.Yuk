using System;
namespace ML.Yuk
{
    public class Window
    {
        NDArray array;

        public Window(NDArray data)
        {
            array = data;
        }

        public bool Equals(Window obj)
        {
            bool valid = true;

            valid = array.Equals(obj.array);

            return valid;
        }

        public double? SumCalc(NDArray data)
        {
            if (data == null)
            {
                return null;
            }

            return NDArray.Sum(data);
        }

        private double? MeanCalc(NDArray data)
        {
            if (data == null)
            {
                return null;
            }

            return NDArray.Mean(data);
        }

        private NDArray Calc(Func<NDArray, double?> f)
        {
            NDArray t = new NDArray();

            foreach (NDArray i in array)
            {
                t.Add(f(i));
            }

            return t;
        }

        public NDArray Mean()
        {
            return Calc(MeanCalc);
        }

        public NDArray Sum()
        {
            return Calc(SumCalc);
        }
    }
}
