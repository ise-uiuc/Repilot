# Repilot Patch

```
            Partial newPartial = new Partial(newTypes, newValues, iChronology);
```

# Developer Patch

```
            Partial newPartial = new Partial(newTypes, newValues, iChronology);
```

# Context

```
--- bug/Time-4/src/main/java/org/joda/time/Partial.java

+++ fix/Time-4/src/main/java/org/joda/time/Partial.java

@@ -461,7 +461,7 @@

             System.arraycopy(iValues, i, newValues, i + 1, newValues.length - i - 1);
             // use public constructor to ensure full validation
             // this isn't overly efficient, but is safe
-            Partial newPartial = new Partial(iChronology, newTypes, newValues);
+            Partial newPartial = new Partial(newTypes, newValues, iChronology);
             iChronology.validate(newPartial, newValues);
             return newPartial;
         }
```

# Note

