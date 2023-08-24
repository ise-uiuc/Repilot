# Repilot Patch

```
Calendar c = new GregorianCalendar(mTimeZone, mLocale);
```

# Developer Patch

```
        Calendar c = new GregorianCalendar(mTimeZone, mLocale);
```

# Context

```
--- bug/Lang-26/src/main/java/org/apache/commons/lang3/time/FastDateFormat.java

+++ fix/Lang-26/src/main/java/org/apache/commons/lang3/time/FastDateFormat.java

@@ -817,7 +817,7 @@

      * @return the formatted string
      */
     public String format(Date date) {
-        Calendar c = new GregorianCalendar(mTimeZone);
+Calendar c = new GregorianCalendar(mTimeZone, mLocale);
         c.setTime(date);
         return applyRules(c, new StringBuffer(mMaxLengthEstimate)).toString();
     }
```

# Note

