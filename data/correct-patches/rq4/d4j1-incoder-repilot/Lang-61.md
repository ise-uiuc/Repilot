# Repilot Patch

```
        int len = size - strLen + 1;
        if (thisBuf == null)
            return len;
```

# Developer Patch

```
        int len = size - strLen + 1;
```

# Context

```
--- bug/Lang-61/src/java/org/apache/commons/lang/text/StrBuilder.java

+++ fix/Lang-61/src/java/org/apache/commons/lang/text/StrBuilder.java

@@ -1773,7 +1773,9 @@

             return -1;
         }
         char[] thisBuf = buffer;
-        int len = thisBuf.length - strLen;
+        int len = size - strLen + 1;
+        if (thisBuf == null)
+            return len;
         outer:
         for (int i = startIndex; i < len; i++) {
             for (int j = 0; j < strLen; j++) {
```

# Note

if (thisBuf == null)
            return len;

`thisBuf` wouldn't be `null` because it's used right after the hunk:

`if (str.charAt(j) != thisBuf[i + j])`