# Repilot Patch

```
next(pos);
```

# Developer Patch

```
next(pos);
```

# Context

```
--- bug/Lang-43/src/java/org/apache/commons/lang/text/ExtendedMessageFormat.java

+++ fix/Lang-43/src/java/org/apache/commons/lang/text/ExtendedMessageFormat.java

@@ -419,6 +419,8 @@

         int start = pos.getIndex();
         char[] c = pattern.toCharArray();
         if (escapingOn && c[start] == QUOTE) {
+
+            next(pos);
             return appendTo == null ? null : appendTo.append(QUOTE);
         }
         int lastHold = start;
```

# Note

