# Repilot Patch

```
        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
```

# Developer Patch

```
        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
```

# Context

```
--- bug/Compress-46/src/main/java/org/apache/commons/compress/archivers/zip/X5455_ExtendedTimestamp.java

+++ fix/Compress-46/src/main/java/org/apache/commons/compress/archivers/zip/X5455_ExtendedTimestamp.java

@@ -526,8 +526,7 @@

     }
 
     private static ZipLong unixTimeToZipLong(long l) {
-        final long TWO_TO_32 = 0x100000000L;
-        if (l >= TWO_TO_32) {
+        if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
             throw new IllegalArgumentException("X5455 timestamps must fit in a signed 32 bit integer: " + l);
         }
         return new ZipLong(l);
```

# Note

