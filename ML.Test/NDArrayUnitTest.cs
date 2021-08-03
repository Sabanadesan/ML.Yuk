using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

using ML.Yuk;
using System.Linq;

namespace ML.Test
{
    public class NDArrayUnitTest
    {
        [Fact]
        public void TestInit()
        {
            NDArray nd = new NDArray();

            nd.Init(5, 3);

            NDArray t = new NDArray(5, 5, 5);

            Assert.True(nd.Equals(t), "Arrays are not equal.");
        }

        [Fact]
        public void TestShapeStr()
        {
            NDArray nd = new NDArray("Car", "Teddy", "Ho");

            NDArray t = nd.Shape();

            Assert.True(t.Equals(new NDArray(3)), "Arrays are not equal.");
        }

        [Fact]
        public void TestShape()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(3, 4, 5);
            NDArray r3 = new NDArray(6, 7, 8);

            NDArray s1 = new NDArray(9, 10, 11);
            NDArray s2 = new NDArray(12, 13, 14);
            NDArray s3 = new NDArray(15, 16, 17);

            NDArray t1 = new NDArray(18, 19, 20);
            NDArray t2 = new NDArray(21, 22, 23);
            NDArray t3 = new NDArray(24, 25, 26);

            NDArray c1 = new NDArray(r1, r2, r3);
            NDArray c2 = new NDArray(s1, s2, s3);
            NDArray c3 = new NDArray(t1, t2, t3);

            NDArray nd = new NDArray(c1, c2, c3);

            NDArray t = nd.Shape();

            Assert.True(t.Equals(new NDArray ( 3, 3, 3 )), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerInt3D()
        {
            NDArray r1 = new NDArray(0, 1);
            NDArray r2 = new NDArray(3, 4);

            NDArray s1 = new NDArray(9, 10);
            NDArray s2 = new NDArray(12, 13);

            NDArray t1 = new NDArray(18, 19);
            NDArray t2 = new NDArray(21, 22);

            NDArray c1 = new NDArray(r1, r2);
            NDArray c2 = new NDArray(s1, s2);
            NDArray c3 = new NDArray(t1, t2);

            NDArray nd = new NDArray(c1, c2, c3);

            int i = nd[1, 1, 1];

            Assert.True(i.Equals(13), "Index is not valid.");
        }

        
        [Fact]
        public void TestIndexerNDArray()
        {

            NDArray r1 = new NDArray(0, 1);
            NDArray r2 = new NDArray(3, 4);

            NDArray s1 = new NDArray(9, 10);
            NDArray s2 = new NDArray(12, 13);

            NDArray t1 = new NDArray(18, 19);
            NDArray t2 = new NDArray(21, 22);

            NDArray c1 = new NDArray(r1, r2);
            NDArray c2 = new NDArray(s1, s2);
            NDArray c3 = new NDArray(t1, t2);

            NDArray nd = new NDArray(c1, c2, c3);

            int i = nd[1, 1, 1];

            NDArray z = new NDArray(0, 1);

            nd[1, 1, 1] = z;

            NDArray j = nd[1, 1, 1];

            Assert.True(j.Equals(new NDArray(0, 1)), "Index is not valid.");
        }

        [Fact]
        public void Test3DEquals()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(3, 4, 5);
            NDArray r3 = new NDArray(6, 7, 8);

            NDArray s1 = new NDArray(9, 10, 11);
            NDArray s2 = new NDArray(12, 13, 14);
            NDArray s3 = new NDArray(15, 16, 17);

            NDArray t1 = new NDArray(18, 19, 20);
            NDArray t2 = new NDArray(21, 22, 23);
            NDArray t3 = new NDArray(24, 25, 26);

            NDArray c1 = new NDArray(r1, r2, r3);
            NDArray c2 = new NDArray(s1, s2, s3);
            NDArray c3 = new NDArray(t1, t2, t3);

            NDArray nd = new NDArray(c1, c2, c3);

            NDArray r11 = new NDArray(0, 1, 2);
            NDArray r22 = new NDArray(3, 4, 5);
            NDArray r33 = new NDArray(6, 7, 8);

            NDArray s11 = new NDArray(9, 10, 11);
            NDArray s22 = new NDArray(12, 13, 14);
            NDArray s33 = new NDArray(15, 16, 17);

            NDArray t11 = new NDArray(18, 19, 20);
            NDArray t22 = new NDArray(21, 22, 23);
            NDArray t33 = new NDArray(24, 25, 26);

            NDArray c11 = new NDArray(r11, r22, r33);
            NDArray c22 = new NDArray(s11, s22, s33);
            NDArray c33 = new NDArray(t11, t22, t33);

            NDArray nd1 = new NDArray(c11, c22, c33);

            Assert.True(nd.Equals(nd1), "Arrays are not equal.");
        }

        [Fact]
        public void TestIntEquals()
        {
            NDArray nd = new NDArray(1, 2, 3);

            NDArray nd1 = new NDArray(1, 2, 3);

            Assert.True(nd.Equals(nd1), "Arrays are not equal.");
        }

        [Fact]
        public void TestIndexerInt()
        {
            NDArray nd = new NDArray(10, 20, 30);

            int t = nd[1];

            Assert.True(t.Equals(20), "Arrays are not equal.");
        }

        [Fact]
        public void TestSlice3D()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(3, 4, 5);
            NDArray r3 = new NDArray(6, 7, 8);

            NDArray s1 = new NDArray(9, 10, 11);
            NDArray s2 = new NDArray(12, 13, 14);
            NDArray s3 = new NDArray(15, 16, 17);

            NDArray t1 = new NDArray(18, 19, 20);
            NDArray t2 = new NDArray(21, 22, 23);
            NDArray t3 = new NDArray(24, 25, 26);

            NDArray c1 = new NDArray(r1, r2, r3);
            NDArray c2 = new NDArray(s1, s2, s3);
            NDArray c3 = new NDArray(t1, t2, t3);

            NDArray nd = new NDArray(c1, c2, c3);

            Slice p1 = new Slice(0, 2);
            Slice p2 = new Slice(1, 2);
            Slice p3 = new Slice(0, 1);

            NDArray t = nd[p1, p2, p3];

            NDArray z1 = new NDArray(0, 1, 2);
            NDArray z2 = new NDArray(3, 4, 5);

            NDArray x1 = new NDArray(12, 13, 14);

            NDArray y1 = new NDArray(18, 19, 20);

            NDArray z = new NDArray(new NDArray(z1, z2), new NDArray(x1), new NDArray(y1));

            Assert.True(t.Equals(z), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceInt()
        {
            NDArray nd = new NDArray(10, 20, 30);

            Slice s = new Slice(0, 1);

            NDArray t = nd[s];
            NDArray t2 = new NDArray(10);

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceStr()
        {
            NDArray nd = new NDArray("Car", "Teddy", "Ho");

            Slice s = new Slice(0, 1);

            NDArray t = nd[s];
            NDArray t2 = new NDArray("Car");

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestCopy()
        {
            NDArray nd = new NDArray("Car", "Teddy", "Ho");

            NDArray nd1 = nd.Copy();

            String s = nd[1];

            nd[1] = "Moo";

            String s1 = nd1[1];

            Assert.True(s1.Equals("Teddy"), "Arrays are not equal.");
        }

        [Fact]
        public void TestDeepCopy()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(3, 4, 5);
            NDArray r3 = new NDArray(6, 7, 8);

            NDArray nd = new NDArray(r1, r2, r3);

            NDArray nd1 = nd.Copy();

            NDArray s = nd[1];

            nd[1] = new NDArray(0, 0, 0);

            NDArray s1 = nd1[1];

            Assert.True(s1.Equals(new NDArray(3, 4, 5)), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceTwoInt()
        {
            NDArray nd = new NDArray(10, 20, 30);

            Slice s = new Slice(0, 1);
            Slice s1 = new Slice(0, 1);

            NDArray t = nd[s, s1];
            NDArray t2 = new NDArray(10 , 10);

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceIregular()
        {
            NDArray nd = new NDArray(10, new NDArray(20), 30);

            Slice s = new Slice(0, 1);
            Slice s1 = new Slice(0, 1);

            NDArray t = nd[s, s1];
            NDArray t2 = new NDArray(10, new NDArray(20));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceNDArray()
        {
            NDArray nd = new NDArray(new NDArray(10, 20, 30));

            Slice s = new Slice(0, 1);

            NDArray t = nd[s];
            NDArray t2 = new NDArray(new NDArray(10));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceTwoNDArray()
        {
            NDArray nd = new NDArray(new NDArray(10, 20), new NDArray(30, 40), new NDArray(50, 60));

            Slice s = new Slice(0, 1);
            Slice s1 = new Slice(1, 2);

            NDArray t = nd[s, s1];
            NDArray t2 = new NDArray(new NDArray(10), new NDArray(40));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceNDArraySingle()
        {
            NDArray nd = new NDArray(new NDArray(10), new NDArray(30), new NDArray(50, 60));

            Slice s = new Slice(0, 1);
            Slice s1 = new Slice(0, 1);

            NDArray t = nd[s, s1];
            NDArray t2 = new NDArray(new NDArray(10), new NDArray(30));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSlice()
        {
            NDArray nd = new NDArray(new NDArray(10, 20, 30), new NDArray(15, 25), new NDArray(1, 2));

            Slice p1 = new Slice(0, 2);
            Slice p2 = new Slice(1, 2);
            Slice p3 = new Slice(0, 1);

            NDArray t = nd[p1, p2, p3];
            NDArray t2 = new NDArray(new NDArray(10, 20), new NDArray(25), new NDArray(1));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestSliceTwoArrays()
        {
            NDArray nd = new NDArray(new NDArray(10, 20, 30), new NDArray(15, 25), new NDArray(1, 2));

            NDArray nd2 = new NDArray(new NDArray(10, 20, 30), new NDArray(15, 25), new NDArray(1, 2));

            Slice p1 = new Slice(0, 2);
            Slice p2 = new Slice(1, 2);
            Slice p3 = new Slice(0, 1);

            NDArray t = nd[p1, p2, p3];
            NDArray t2 = nd2[p1, p2, p3];

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRange()
        {
            NDArray nd = new NDArray(new NDArray(10, 20, 30), new NDArray(15, 25), new NDArray(1, 2));

            Range p3 = new Range(0, 1);

            NDArray t = nd[.., 0..1, p3];
            NDArray t2 = new NDArray(new NDArray(10, 20, 30), new NDArray(15), new NDArray(1));

            Assert.True(t.Equals(t2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRemove()
        {
            NDArray nd = new NDArray(1, 2, 3);

            nd.Remove(2);

            NDArray nd1 = new NDArray(1, 2);

            Assert.True(nd.Equals(nd1), "Arrays are not equal.");
        }

        [Fact]
        public void TestStd()
        {
            NDArray nd = new NDArray(1, 2, 3);

            double i = NDArray.Std(nd);

            Assert.True(i.Equals(0.816496580927726), "Arrays are not equal.");
        }

        [Fact]
        public void TestMean()
        {
            NDArray nd = new NDArray(1, 2, 3);

            double i = NDArray.Mean(nd);

            Assert.True(i.Equals(2), "Arrays are not equal.");
        }

        [Fact]
        public void TestMeanNull()
        {
            NDArray nd = new NDArray(1, null, 3);

            double i = NDArray.Mean(nd);

            Assert.True(i.Equals(1.3333333333333333), "Arrays are not equal.");
        }

        [Fact]
        public void TestStdNull()
        {
            NDArray nd = new NDArray(1, null, 3);

            double i = NDArray.Std(nd);

            Assert.True(i.Equals(0.98130676292531638), "Arrays are not equal.");
        }

        [Fact]
        public void TestAsType()
        {
            NDArray nd = new NDArray(1, 2, 3);

            int[] i = nd.AsType<int>();

            int[] t = new int[] { 1, 2, 3 };

            Assert.True(Enumerable.SequenceEqual(i, t), "Arrays are not equal.");
        }

        [Fact]
        public void TestAsTypeArray()
        {
            NDArray nd = new NDArray(new dynamic[] { 1, 2, 3 });

            int[] i = nd.AsType<int>();

            int[] t = new int[] { 1, 2, 3 };

            Assert.True(Enumerable.SequenceEqual(i, t), "Arrays are not equal.");
        }

        [Fact]
        public void TestAsTypeArrayInt()
        {
            NDArray nd = new NDArray(new int[] { 1, 2, 3 });

            int[] i = nd.AsType<int>();

            int[] t = new int[] { 1, 2, 3 };

            Assert.True(Enumerable.SequenceEqual(i, t), "Arrays are not equal.");
        }

        [Fact]
        public void TestGetRows()
        {
            NDArray nd = new NDArray(1, 2, 3);

            dynamic i = null;

            foreach (dynamic r in nd)
            {
                i = r;
            }

            Assert.True(i.Equals(3), "Arrays are not equal.");
        }

        [Fact]
        public void TestRolling2()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(2);

            NDArray nd2 = new NDArray(null, new NDArray(1, 2), new NDArray(2, 3), new NDArray(3, 4), new NDArray(4, 5), new NDArray(5, 6));

            Window w2 = new Window(nd2);

            Assert.True(w1.Equals(w2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRolling3()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(3);

            NDArray nd2 = new NDArray(null, null, new NDArray(1, 2, 3), new NDArray(2, 3, 4), new NDArray(3, 4, 5), new NDArray(4, 5, 6));

            Window w2 = new Window(nd2);

            Assert.True(w1.Equals(w2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRollingSum2()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(2);

            NDArray o = w1.Sum();

            NDArray nd2 = new NDArray(null, 3, 5, 7, 9, 11);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRollingSum3()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(3);

            NDArray o = w1.Sum();

            NDArray nd2 = new NDArray(null, null, 6, 9, 12, 15);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRollingMean2()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(2);

            NDArray o = w1.Mean();

            NDArray nd2 = new NDArray(null, 1.5, 2.5, 3.5, 4.5, 5.5);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

        [Fact]
        public void TestRollingMean3()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Window w1 = nd.Rolling(3);

            NDArray o = w1.Mean();

            NDArray nd2 = new NDArray(null, null, 2, 3, 4, 5);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

        [Fact]
        public void TestContains1()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Assert.True(nd.Contains(5), "Arrays are not equal.");
        }

        [Fact]
        public void TestContains2()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            Assert.False(nd.Contains(9), "Arrays are not equal.");
        }

        [Fact]
        public void TestFindIndex1()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            int i = nd.FindIndex(3);

            Assert.True(nd.Contains(5), "Arrays are not equal.");
        }

        [Fact]
        public void TestFindIndex2()
        {
            NDArray nd = new NDArray(1, 2, 3, 4, 5, 6);

            int i = nd.FindIndex(8);

            Assert.True(i.Equals(-1), "Arrays are not equal.");
        }

        [Fact]
        public void TestAppend1()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(3, 4, 5);

            NDArray r3 = r1.Append(r2);

            NDArray r4 = new NDArray(0, 1, 2, 3, 4, 5);


            Assert.True(r3.Equals(r4), "Arrays are not equal.");
        }

        [Fact]
        public void TestAppend2()
        {
            NDArray r1 = new NDArray(0, 1, 2);
            NDArray r2 = new NDArray(1, 0, 5);

            NDArray r3 = r1.Append(r2);

            NDArray r4 = new NDArray(0, 1, 2, 5);


            Assert.True(r3.Equals(r4), "Arrays are not equal.");
        }

        [Fact]
        public void TestMin()
        {
            NDArray nd = new NDArray(null, 8, 2, -10, 4, -6);

            double i = NDArray.Min(nd);

            Assert.True(i.Equals(-10), "Arrays are not equal.");
        }

        [Fact]
        public void TestMax()
        {
            NDArray nd = new NDArray(null, 8, 2, -10, 4, -6);

            double i = NDArray.Max(nd);

            Assert.True(i.Equals(8), "Arrays are not equal.");
        }

        [Fact]
        public void TestLocalMax()
        {
            NDArray nd = new NDArray(null, 8, 2, -10, 4, -6);

            NDArray o = nd.Maxlextrema(2);

            NDArray nd2 = new NDArray(null, 8, null, null, 4, null);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

        [Fact]
        public void TestLocalMin()
        {
            NDArray nd = new NDArray(null, 8, 2, -10, 4, -6);

            NDArray o = nd.Minlextrema(2);

            NDArray nd2 = new NDArray(null, null, null, -10, null, null);

            Assert.True(o.Equals(nd2), "Arrays are not equal.");
        }

    }
}
