# Repilot Patch

```
return StringUtils.newStringUtf8(encodeBase64(binaryData, false));
```

# Developer Patch

```
        return StringUtils.newStringUtf8(encodeBase64(binaryData, false));
```

# Context

```
--- bug/Codec-7/src/java/org/apache/commons/codec/binary/Base64.java

+++ fix/Codec-7/src/java/org/apache/commons/codec/binary/Base64.java

@@ -667,7 +667,7 @@

      * @since 1.4
      */    
     public static String encodeBase64String(byte[] binaryData) {
-        return StringUtils.newStringUtf8(encodeBase64(binaryData, true));
+return StringUtils.newStringUtf8(encodeBase64(binaryData, false));
     }
     
     /**
```

# Note

