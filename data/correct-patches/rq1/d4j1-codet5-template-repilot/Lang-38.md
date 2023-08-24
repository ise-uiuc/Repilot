# Repilot Patch

```

            Date date = calendar.getTime();
```

# Developer Patch

```
            calendar.getTime(); /// LANG-538
```

# Context

```
--- bug/Lang-38/src/java/org/apache/commons/lang3/time/FastDateFormat.java

+++ fix/Lang-38/src/java/org/apache/commons/lang3/time/FastDateFormat.java

@@ -869,6 +869,8 @@

      */
     public StringBuffer format(Calendar calendar, StringBuffer buf) {
         if (mTimeZoneForced) {
+
+            Date date = calendar.getTime();
             calendar = (Calendar) calendar.clone();
             calendar.setTimeZone(mTimeZone);
         }
```

# Note

